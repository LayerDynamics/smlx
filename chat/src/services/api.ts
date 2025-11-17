// SMLX API Client (OpenAI-compatible)

import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ModelList,
  APIError,
} from '../types/api';

export class SMLXAPIError extends Error {
  statusCode: number;
  type?: string;

  constructor(statusCode: number, message: string, type?: string) {
    super(message);
    this.name = 'SMLXAPIError';
    this.statusCode = statusCode;
    this.type = type;
  }
}

export class SMLXClient {
  private baseURL: string;
  private headers: HeadersInit;

  constructor(baseURL: string = 'http://localhost:8000/v1') {
    this.baseURL = baseURL;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      let errorType = 'api_error';

      try {
        const errorData: APIError = await response.json();
        errorMessage = errorData.error.message;
        errorType = errorData.error.type;
      } catch {
        // If parsing fails, use default error message
      }

      throw new SMLXAPIError(response.status, errorMessage, errorType);
    }

    return response.json();
  }

  // Chat completions (non-streaming)
  async createChatCompletion(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request),
    });

    return this.handleResponse<ChatCompletionResponse>(response);
  }

  // Chat completions (streaming via SSE)
  async *streamChatCompletion(
    request: ChatCompletionRequest
  ): AsyncGenerator<ChatCompletionChunk, void, unknown> {
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new SMLXAPIError(
        response.status,
        `Failed to start streaming: ${response.statusText}`
      );
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();

          if (!trimmedLine || trimmedLine === 'data: [DONE]') {
            continue;
          }

          if (trimmedLine.startsWith('data: ')) {
            const data = trimmedLine.slice(6);

            try {
              const chunk: ChatCompletionChunk = JSON.parse(data);
              yield chunk;
            } catch (e) {
              console.error('Failed to parse SSE data:', data, e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // List models
  async listModels(): Promise<ModelList> {
    const response = await fetch(`${this.baseURL}/models`, {
      method: 'GET',
      headers: this.headers,
    });

    return this.handleResponse<ModelList>(response);
  }

  // Get model info
  async getModel(modelId: string) {
    const response = await fetch(`${this.baseURL}/models/${modelId}`, {
      method: 'GET',
      headers: this.headers,
    });

    return this.handleResponse(response);
  }

  // Unload model (custom endpoint)
  async unloadModel(modelId: string): Promise<void> {
    const response = await fetch(`${this.baseURL}/models/${modelId}`, {
      method: 'DELETE',
      headers: this.headers,
    });

    await this.handleResponse(response);
  }

  // Health check
  async healthCheck(): Promise<{
    status: string;
    timestamp: number;
    models_loaded: string[];
  }> {
    const response = await fetch(`${this.baseURL.replace('/v1', '')}/health`, {
      method: 'GET',
      headers: this.headers,
    });

    return this.handleResponse(response);
  }
}

// Export singleton instance
export const smlxClient = new SMLXClient();
