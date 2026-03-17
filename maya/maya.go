package maya

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Maya 客户端，封装与 Python 推理服务的 HTTP 通信
type Maya struct {
	BaseURL string
	Client  *http.Client
}

// ChatRequest 对话请求
type ChatRequest struct {
	Text     string `json:"text"`
	MaxToken int    `json:"max_new_tokens"`
}

// ChatResponse 对话响应
type ChatResponse struct {
	Text string `json:"text"`
}

// TTSRequest 语音合成请求
type TTSRequest struct {
	Text        string `json:"text"`
	Description string `json:"description,omitempty"`
	MaxToken    int    `json:"max_new_tokens,omitempty"`
}

// TTSResponse 语音合成响应
type TTSResponse struct {
	AudioBase64     string  `json:"audio_base64"`
	SampleRate      int     `json:"sample_rate"`
	DurationSeconds float64 `json:"duration_seconds"`
}

// NewMaya 创建 Maya 客户端实例
func NewMaya(baseURL string) *Maya {
	return &Maya{
		BaseURL: baseURL,
		Client:  &http.Client{},
	}
}

// HealthCheck 检查 Python 推理服务是否就绪
func (m *Maya) HealthCheck() error {
	resp, err := m.Client.Get(m.BaseURL + "/health")
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}
	return nil
}

// Chat 发送对话请求，返回模型生成的文本
func (m *Maya) Chat(text string, maxTokens int) (*ChatResponse, error) {
	req := ChatRequest{
		Text:     text,
		MaxToken: maxTokens,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}

	resp, err := m.Client.Post(m.BaseURL+"/chat", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("post chat request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("chat returned status %d: %s", resp.StatusCode, string(errBody))
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read chat response: %w", err)
	}

	var result ChatResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("unmarshal chat response: %w", err)
	}

	return &result, nil
}

// TTS 发送语音合成请求，返回 Base64 编码的音频数据
func (m *Maya) TTS(text, description string) (*TTSResponse, error) {
	req := TTSRequest{
		Text:        text,
		Description: description,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal tts request: %w", err)
	}

	resp, err := m.Client.Post(m.BaseURL+"/tts", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("post tts request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("tts returned status %d: %s", resp.StatusCode, string(errBody))
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read tts response: %w", err)
	}

	var result TTSResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("unmarshal tts response: %w", err)
	}

	return &result, nil
}
