package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
)

// -----------------------------
// Python 服务配置
// -----------------------------
var (
	PythonExe      = filepath.Join("runtime", "python", "3.11.9", "python.exe")
	SitePackages   = filepath.Join("runtime", "python", "3.11.9", "site-packages")
	Requirements   = "requirements.txt"
	ServerPath     = "maya1_server.py"
	ServerURL      = "http://127.0.0.1:5005"
	ModelDir       = filepath.Join("runtime", "python", "models", "maya1")
	DownloadScript = "download_maya1.py"
)

// -----------------------------
// 检查模型是否已下载
// -----------------------------
func IsModelReady() bool {
	requiredFiles := []string{
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"model.safetensors.index.json",
	}
	for _, f := range requiredFiles {
		if _, err := os.Stat(filepath.Join(ModelDir, f)); os.IsNotExist(err) {
			return false
		}
	}

	// 检查是否有 safetensors 权重文件
	entries, err := os.ReadDir(ModelDir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if len(name) > len("model-") && name[:6] == "model-" &&
			filepath.Ext(name) == ".safetensors" {
			return true
		}
	}
	return false
}

// -----------------------------
// 下载模型
// -----------------------------
func DownloadModel() error {
	fmt.Println("========================================")
	fmt.Println("  Maya1 模型未找到，开始下载...")
	fmt.Println("  模型大小约 6GB，请耐心等待")
	fmt.Println("========================================")

	cmd := exec.Command(PythonExe, DownloadScript)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("模型下载失败: %w", err)
	}

	if !IsModelReady() {
		return fmt.Errorf("下载完成但模型文件不完整，请重试")
	}

	fmt.Println("模型下载完成！")
	return nil
}

// -----------------------------
// 安装 Python 依赖
// -----------------------------
func IsDepsInstalled() bool {
	// 检查 transformers 包是否存在作为标志
	_, err := os.Stat(filepath.Join(SitePackages, "transformers"))
	return err == nil
}

func InstallDeps() error {
	fmt.Println("========================================")
	fmt.Println("  安装 Python 依赖...")
	fmt.Println("========================================")

	// 先检查 pip 是否可用，不可用则用 get-pip.py 引导安装
	checkPip := exec.Command(PythonExe, "-m", "pip", "--version")
	if err := checkPip.Run(); err != nil {
		fmt.Println("pip 未安装，正在引导安装...")

		getPipPath := filepath.Join("runtime", "python", "get-pip.py")

		// 如果 get-pip.py 不存在，从网上下载
		if _, err := os.Stat(getPipPath); os.IsNotExist(err) {
			fmt.Println("下载 get-pip.py...")
			resp, err := http.Get("https://bootstrap.pypa.io/get-pip.py")
			if err != nil {
				return fmt.Errorf("下载 get-pip.py 失败: %w", err)
			}
			defer resp.Body.Close()

			data, err := io.ReadAll(resp.Body)
			if err != nil {
				return fmt.Errorf("读取 get-pip.py 失败: %w", err)
			}

			if err := os.MkdirAll(filepath.Dir(getPipPath), 0755); err != nil {
				return fmt.Errorf("创建目录失败: %w", err)
			}
			if err := os.WriteFile(getPipPath, data, 0644); err != nil {
				return fmt.Errorf("保存 get-pip.py 失败: %w", err)
			}
		}

		pipInstall := exec.Command(PythonExe, getPipPath)
		pipInstall.Stdout = os.Stdout
		pipInstall.Stderr = os.Stderr
		if err := pipInstall.Run(); err != nil {
			return fmt.Errorf("pip 引导安装失败: %w", err)
		}
		fmt.Println("pip 安装成功！")
	}

	fmt.Println("正在安装依赖包（可能需要几分钟）...")
	cmd := exec.Command(PythonExe, "-m", "pip", "install",
		"--target", SitePackages,
		"-r", Requirements,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("依赖安装失败: %w", err)
	}

	fmt.Println("Python 依赖安装完成！")
	return nil
}

// -----------------------------
// 启动 Python 服务
// -----------------------------
func StartPythonServer() (*exec.Cmd, error) {
	cmd := exec.Command(PythonExe, ServerPath)

	// 不弹出黑窗口（Windows）
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}

	// 输出重定向
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	fmt.Println("启动 Python 服务中...")

	err := cmd.Start()
	if err != nil {
		return nil, err
	}

	// 等待服务启动（模型加载可能需要数分钟）
	for i := 0; i < 300; i++ {
		resp, err := http.Get(ServerURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				fmt.Println("Python 服务已启动成功！")
				return cmd, nil
			}
		}
		time.Sleep(1 * time.Second)
	}

	return nil, fmt.Errorf("Python 服务启动超时")
}

// -----------------------------
// Chat 请求结构
// -----------------------------
type ChatRequest struct {
	Text     string `json:"text"`
	MaxToken int    `json:"max_new_tokens"`
}

type ChatResponse struct {
	Text string `json:"text"`
}

// -----------------------------
// TTS 请求结构
// -----------------------------
type TTSRequest struct {
	Text        string `json:"text"`
	Description string `json:"description,omitempty"`
	MaxToken    int    `json:"max_new_tokens,omitempty"`
}

type TTSResponse struct {
	AudioBase64     string  `json:"audio_base64"`
	SampleRate      int     `json:"sample_rate"`
	DurationSeconds float64 `json:"duration_seconds"`
}

// -----------------------------
// 测试 Chat
// -----------------------------
func TestChat() {
	req := ChatRequest{
		Text:     "你好，介绍一下你自己",
		MaxToken: 200,
	}

	body, err := json.Marshal(req)
	if err != nil {
		fmt.Println("序列化请求失败:", err)
		return
	}

	resp, err := http.Post(ServerURL+"/chat", "application/json", bytes.NewBuffer(body))
	if err != nil {
		fmt.Println("请求失败:", err)
		return
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("读取响应失败:", err)
		return
	}

	var result ChatResponse
	if err := json.Unmarshal(data, &result); err != nil {
		fmt.Println("解析响应失败:", err)
		return
	}

	fmt.Println("Chat 返回：", result.Text)
}

// -----------------------------
// 测试 TTS
// -----------------------------
func TestTTS() {
	log.Println("测试 TTS 接口...")
	req := TTSRequest{
		Text: "Hello!",
		Description: "Realistic female voice in the 30s age with american accent. " +
			"Normal pitch, warm timbre, conversational pacing, neutral tone delivery at med intensity.",
		MaxToken: 200,
	}
	log.Printf("请求文本：%s\n", req.Text)
	body, err := json.Marshal(req)
	if err != nil {
		fmt.Println("序列化请求失败:", err)
		return
	}
	log.Printf("请求 JSON：%s\n", string(body))

	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Post(ServerURL+"/tts", "application/json", bytes.NewBuffer(body))
	if err != nil {
		fmt.Println("请求失败:", err)
		return
	}
	defer resp.Body.Close()
	log.Printf("响应状态码：%d\n", resp.StatusCode)
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("读取响应失败:", err)
		return
	}

	var result TTSResponse
	if err := json.Unmarshal(data, &result); err != nil {
		fmt.Println("解析响应失败:", err)
		return
	}
	log.Printf("响应 JSON：%s\n", string(data))
	fmt.Println("TTS Base64 长度：", len(result.AudioBase64))

	// 保存为 wav 文件
	audioBytes, err := base64.StdEncoding.DecodeString(result.AudioBase64)
	if err != nil {
		fmt.Println("Base64 解码失败:", err)
		return
	}

	if err := os.WriteFile("output.wav", audioBytes, 0644); err != nil {
		fmt.Println("保存文件失败:", err)
		return
	}

	fmt.Println("已保存到 output.wav")
}

// -----------------------------
// 主入口
// -----------------------------
func main() {
	// 检查并安装 Python 依赖
	if !IsDepsInstalled() {
		if err := InstallDeps(); err != nil {
			fmt.Println(err)
			return
		}
	} else {
		fmt.Println("Python 依赖已就绪")
	}

	// 检查并下载模型
	if !IsModelReady() {
		if err := DownloadModel(); err != nil {
			fmt.Println(err)
			return
		}
	} else {
		fmt.Println("模型已就绪:", ModelDir)
	}

	// 启动 Python 服务
	cmd, err := StartPythonServer()
	if err != nil {
		fmt.Println("Python 服务启动失败:", err)
		return
	}
	defer cmd.Process.Kill()

	// 测试 Chat
	// TestChat()

	// 测试 TTS
	TestTTS()
}
