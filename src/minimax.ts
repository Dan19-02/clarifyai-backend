import axios from 'axios';
import readline from 'readline';
import { Readable } from 'stream';

export interface MinimaxOptions {
  model?: string;
  messages?: any[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
}

export async function callMinimax(opts: MinimaxOptions = {}) {
  const API_URL = process.env.MINIMAX_API_URL || 'https://integrate.api.nvidia.com/v1/chat/completions';
  const API_KEY = process.env.MINIMAX_API_KEY;
  if (!API_KEY) throw new Error('MINIMAX_API_KEY environment variable is required');

  const payload = {
    model: opts.model ?? 'minimaxai/minimax-m3',
    messages: opts.messages ?? [{ role: 'user', content: '' }],
    max_tokens: opts.max_tokens ?? 8192,
    temperature: opts.temperature ?? 1.0,
    top_p: opts.top_p ?? 0.95,
    stream: !!opts.stream,
  };

  const headers: Record<string, string> = {
    Authorization: `Bearer ${API_KEY}`,
    Accept: opts.stream ? 'text/event-stream' : 'application/json',
    'Content-Type': 'application/json',
  };

  const res = await axios.post(API_URL, payload, {
    headers,
    responseType: opts.stream ? 'stream' : 'json',
  });

  if (opts.stream) {
    return res.data as Readable;
  }

  return res.data;
}

export function streamToLines(stream: Readable, onLine: (line: string) => void) {
  const rl = readline.createInterface({ input: stream });
  rl.on('line', onLine);
  return rl;
}

export default { callMinimax, streamToLines };
