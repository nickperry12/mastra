import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { Context } from 'hono';
import {
  getAgentsHandler,
  getAgentByIdHandler,
  getEvalsByAgentIdHandler,
  getLiveEvalsByAgentIdHandler,
} from '../agents';
import { EvalRow } from '@mastra/core/storage';

// Mock the original handlers from @mastra/server/handlers/agents
vi.mock('@mastra/server/handlers/agents', () => ({
  getAgentsHandler: vi.fn(),
  getAgentByIdHandler: vi.fn(),
  getEvalsByAgentIdHandler: vi.fn(),
  getLiveEvalsByAgentIdHandler: vi.fn(),
}));

// Import after mocking
import {
  getAgentsHandler as getOriginalAgentsHandler,
  getAgentByIdHandler as getOriginalAgentByIdHandler,
  getEvalsByAgentIdHandler as getOriginalEvalsByAgentIdHandler,
  getLiveEvalsByAgentIdHandler as getOriginalLiveEvalsByAgentIdHandler,
} from '@mastra/server/handlers/agents';

describe('getAgentsHandler', () => {
  let mockContext: Partial<Context>;
  let mockMastra: any;
  let mockRuntimeContext: any;

  beforeEach(() => {
    vi.clearAllMocks();
    mockMastra = { id: 'mock-mastra' };
    mockRuntimeContext = { foo: 'bar' };
    mockContext = {
      get: vi.fn((key: string) => {
        if (key === 'mastra') return mockMastra;
        if (key === 'runtimeContext') return mockRuntimeContext;
        return undefined;
      }),
      json: vi.fn(
        (data, status) =>
          new Response(JSON.stringify(data), {
            status: status || 200,
            headers: { 'Content-Type': 'application/json' },
          }),
      ) as any,
    };
  });

  it('should return serialized agents as JSON with status 200', async () => {
    const agentsList = {
      agent1: {
        id: 'agent1',
        name: 'Agent One',
        instructions: 'Do something',
        tools: [],
        workflows: {},
        provider: 'test',
        modelId: 'model-1',
        defaultGenerateOptions: {},
        defaultStreamOptions: {},
      },
      agent2: {
        id: 'agent2',
        name: 'Agent Two',
        instructions: 'Do something else',
        tools: [],
        workflows: {},
        provider: 'test',
        modelId: 'model-2',
        defaultGenerateOptions: {},
        defaultStreamOptions: {},
      },
    };
    vi.mocked(getOriginalAgentsHandler).mockResolvedValue(agentsList);

    const result = await getAgentsHandler(mockContext as Context);

    expect(getOriginalAgentsHandler).toHaveBeenCalledWith({
      mastra: mockMastra,
      runtimeContext: mockRuntimeContext,
    });
    expect(mockContext.json).toHaveBeenCalledWith(agentsList);
    expect(result).toBeInstanceOf(Response);
    expect(result.status).toBe(200);
    const json = await result.json();
    expect(json).toEqual(agentsList);
  });
});

describe('getAgentByIdHandler', () => {
  let mockContext: Partial<Context>;
  let mockMastra: any;
  let mockRuntimeContext: any;
  const agentId = 'agent1';

  beforeEach(() => {
    vi.clearAllMocks();
    mockMastra = { id: 'mock-mastra' };
    mockRuntimeContext = { foo: 'bar' };
    mockContext = {
      get: vi.fn((key: string) => {
        if (key === 'mastra') return mockMastra;
        if (key === 'runtimeContext') return mockRuntimeContext;
        if (key === 'playground') return false;
        return undefined;
      }),
      req: {
        param: vi.fn((key: string) => (key === 'agentId' ? agentId : undefined)),
        header: vi.fn(() => undefined),
      } as any,
      json: vi.fn(
        (data, status) =>
          new Response(JSON.stringify(data), {
            status: status || 200,
            headers: { 'Content-Type': 'application/json' },
          }),
      ) as any,
    };
  });

  it('should return agent details as JSON with status 200', async () => {
    const agentDetails = {
      id: agentId,
      name: 'Test Agent',
      instructions: 'Test instructions',
      tools: [],
      workflows: {},
      provider: 'test',
      modelId: 'model-1',
      defaultGenerateOptions: {},
      defaultStreamOptions: {},
    };
    vi.mocked(getOriginalAgentByIdHandler).mockResolvedValue(agentDetails);

    const result = await getAgentByIdHandler(mockContext as Context);

    expect(getOriginalAgentByIdHandler).toHaveBeenCalledWith({
      mastra: mockMastra,
      agentId,
      runtimeContext: mockRuntimeContext,
      isPlayground: false,
    });
    expect(mockContext.json).toHaveBeenCalledWith(agentDetails);
    expect(result).toBeInstanceOf(Response);
    expect(result.status).toBe(200);
    const json = await result.json();
    expect(json).toEqual(agentDetails);
  });
});

describe('getEvalsByAgentIdHandler', () => {
  let mockContext: Partial<Context>;
  let mockMastra: any;
  let mockRuntimeContext: any;
  const agentId = 'agent1';

  beforeEach(() => {
    vi.clearAllMocks();
    mockMastra = { id: 'mock-mastra' };
    mockRuntimeContext = { foo: 'bar' };
    mockContext = {
      get: vi.fn((key: string) => {
        if (key === 'mastra') return mockMastra;
        if (key === 'runtimeContext') return mockRuntimeContext;
        return undefined;
      }),
      req: {
        param: vi.fn((key: string) => (key === 'agentId' ? agentId : undefined)),
      } as any,
      json: vi.fn(
        (data, status) =>
          new Response(JSON.stringify(data), {
            status: status || 200,
            headers: { 'Content-Type': 'application/json' },
          }),
      ) as any,
    };
  });

  it('should return evals as JSON with status 200', async () => {
    const evals: EvalRow[] = [
      {
        input: 'test input',
        output: 'test output',
        result: { score: 0.9 },
        agentName: 'agent1',
        createdAt: '2024-01-01T00:00:00Z',
        metricName: 'accuracy',
        instructions: 'Eval instructions',
        runId: 'run1',
        globalRunId: 'globalRun1',
      },
    ];

    const value = {
      id: 'eval1',
      name: 'Eval One',
      instructions: 'Eval instructions',
      evals: evals,
    };

    vi.mocked(getOriginalEvalsByAgentIdHandler).mockResolvedValue(value);

    const result = await getEvalsByAgentIdHandler(mockContext as Context);

    expect(getOriginalEvalsByAgentIdHandler).toHaveBeenCalledWith({
      mastra: mockMastra,
      agentId,
      runtimeContext: mockRuntimeContext,
    });
    expect(mockContext.json).toHaveBeenCalledWith(value);
    expect(result).toBeInstanceOf(Response);
    expect(result.status).toBe(200);
    const json = await result.json();
    console.log('JSON:', json);
    expect(json).toEqual(value);
  });
});

describe('getLiveEvalsByAgentIdHandler', () => {
  let mockContext: Partial<Context>;
  let mockMastra: any;
  let mockRuntimeContext: any;
  const agentId = 'agent1';

  beforeEach(() => {
    vi.clearAllMocks();
    mockMastra = { id: 'mock-mastra' };
    mockRuntimeContext = { foo: 'bar' };
    mockContext = {
      get: vi.fn((key: string) => {
        if (key === 'mastra') return mockMastra;
        if (key === 'runtimeContext') return mockRuntimeContext;
        return undefined;
      }),
      req: {
        param: vi.fn((key: string) => (key === 'agentId' ? agentId : undefined)),
      } as any,
      json: vi.fn(
        (data, status) =>
          new Response(JSON.stringify(data), {
            status: status || 200,
            headers: { 'Content-Type': 'application/json' },
          }),
      ) as any,
    };
  });

  it('should return live evals as JSON with status 200', async () => {
    const liveEvals: EvalRow[] = [
      {
        input: 'test input',
        output: 'test output',
        result: { score: 0.9 },
        agentName: 'agent1',
        createdAt: '2024-01-01T00:00:00Z',
        metricName: 'accuracy',
        instructions: 'Eval instructions',
        runId: 'run1',
        globalRunId: 'globalRun1',
      },
    ];

    const value = {
      id: 'eval1',
      name: 'Eval One',
      instructions: 'Eval instructions',
      evals: liveEvals,
    };
    vi.mocked(getOriginalLiveEvalsByAgentIdHandler).mockResolvedValue(value);

    const result = await getLiveEvalsByAgentIdHandler(mockContext as Context);

    expect(getOriginalLiveEvalsByAgentIdHandler).toHaveBeenCalledWith({
      mastra: mockMastra,
      agentId,
      runtimeContext: mockRuntimeContext,
    });
    expect(mockContext.json).toHaveBeenCalledWith(value);
    expect(result).toBeInstanceOf(Response);
    expect(result.status).toBe(200);
    const json = await result.json();
    expect(json).toEqual(value);
  });
});
