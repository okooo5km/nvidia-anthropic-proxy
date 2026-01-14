const API_BASE = 'https://integrate.api.nvidia.com/v1';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key, anthropic-version',
        },
      });
    }

    if (env.AUTH_TOKEN) {
      const auth = request.headers.get('x-api-key') || request.headers.get('Authorization')?.replace('Bearer ', '');
      if (auth !== env.AUTH_TOKEN) {
        return json({ error: { type: 'authentication_error', message: 'Invalid API key' } }, 401);
      }
    }

    if (url.pathname === '/v1/messages' && request.method === 'POST') {
      return handleMessages(request, env);
    }
    if (url.pathname === '/v1/models' && request.method === 'GET') {
      return handleModels();
    }
    if (url.pathname === '/health' || url.pathname === '/') {
      return json({ status: 'ok' });
    }

    return json({ error: { type: 'not_found', message: 'Not found' } }, 404);
  },
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
  });
}

function handleModels() {
  return json({ data: [], has_more: false, first_id: null, last_id: null });
}

async function handleMessages(request, env) {
  try {
    const body = await request.json();
    if (!body.model) {
      return json({ error: { type: 'invalid_request_error', message: 'model is required' } }, 400);
    }

    const messages = [];
    if (body.system) messages.push({ role: 'system', content: body.system });
    for (const msg of body.messages) {
      messages.push({ role: msg.role, content: convertContent(msg.content) });
    }

    const payload = {
      model: body.model,
      messages,
      max_tokens: body.max_tokens,
      stream: !!body.stream,
    };
    if (body.temperature !== undefined) payload.temperature = body.temperature;
    if (body.top_p !== undefined) payload.top_p = body.top_p;
    if (body.stop_sequences) payload.stop = body.stop_sequences;

    const res = await fetch(`${API_BASE}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${env.NVIDIA_API_KEY}`,
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) return json({ error: { type: 'api_error', message: await res.text() } }, res.status);

    if (body.stream) return handleStream(res, body.model);

    const data = await res.json();
    const choice = data.choices?.[0];
    return json({
      id: data.id || `msg_${Date.now()}`,
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: choice?.message?.content || '' }],
      model: body.model,
      stop_reason: choice?.finish_reason === 'length' ? 'max_tokens' : 'end_turn',
      usage: {
        input_tokens: data.usage?.prompt_tokens || 0,
        output_tokens: data.usage?.completion_tokens || 0,
      },
    });
  } catch (err) {
    return json({ error: { type: 'internal_error', message: err.message } }, 500);
  }
}

function convertContent(content) {
  if (typeof content === 'string') return content;
  return content.map(block => {
    if (block.type === 'text') return { type: 'text', text: block.text };
    if (block.type === 'image') {
      return { type: 'image_url', image_url: { url: `data:${block.source.media_type};base64,${block.source.data}` } };
    }
    return block;
  });
}

function handleStream(response, model) {
  const id = `msg_${Date.now()}`;
  const enc = new TextEncoder();
  let tokens = 0;

  const readable = new ReadableStream({
    async start(ctrl) {
      const send = (event, data) => ctrl.enqueue(enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));

      send('message_start', {
        type: 'message_start',
        message: { id, type: 'message', role: 'assistant', content: [], model, stop_reason: null, usage: { input_tokens: 0, output_tokens: 0 } },
      });
      send('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });

      const es = EventSource.from(response.body);
      es.onmessage = (e) => {
        if (e.data === '[DONE]') return;
        try {
          const parsed = JSON.parse(e.data);
          const delta = parsed.choices?.[0]?.delta;
          const finish = parsed.choices?.[0]?.finish_reason;

          if (delta?.content) {
            send('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta.content } });
          }
          if (finish) {
            send('content_block_stop', { type: 'content_block_stop', index: 0 });
            send('message_delta', { type: 'message_delta', delta: { stop_reason: finish === 'length' ? 'max_tokens' : 'end_turn' }, usage: { output_tokens: tokens } });
            send('message_stop', { type: 'message_stop' });
            ctrl.close();
          }
          if (parsed.usage) tokens = parsed.usage.completion_tokens || 0;
        } catch {}
      };
      es.onerror = () => ctrl.close();
    },
  });

  return new Response(readable, {
    headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*' },
  });
}
