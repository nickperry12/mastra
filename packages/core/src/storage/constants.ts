import type { StorageColumn } from './types';

export const TABLE_WORKFLOW_SNAPSHOT = 'mastra_workflow_snapshot';
export const TABLE_EVALS = 'mastra_evals';
export const TABLE_MESSAGES = 'mastra_messages';
export const TABLE_THREADS = 'mastra_threads';
export const TABLE_TRACES = 'mastra_traces';
export const TABLE_RESOURCES = 'mastra_resources';
export const TABLE_SCORERS = 'mastra_scorers';

export type TABLE_NAMES =
  | typeof TABLE_WORKFLOW_SNAPSHOT
  | typeof TABLE_EVALS
  | typeof TABLE_MESSAGES
  | typeof TABLE_THREADS
  | typeof TABLE_TRACES
  | typeof TABLE_RESOURCES
  | typeof TABLE_SCORERS;

export const SCORERS_SCHEMA: Record<string, StorageColumn> = {
  id: { type: 'text', nullable: false, primaryKey: true },
  scorerId: {
    type: 'text',
  },
  traceId: {
    type: 'text',
    nullable: true,
  },
  runId: {
    type: 'text',
  },
  scorer: {
    type: 'jsonb',
  },
  extractStepResult: {
    type: 'jsonb',
    nullable: true,
  },
  analyzeStepResult: {
    type: 'jsonb',
    nullable: true,
  },
  score: {
    type: 'float',
  },
  reason: {
    type: 'text',
    nullable: true,
  },
  metadata: {
    type: 'jsonb',
    nullable: true,
  },
  extractPrompt: {
    type: 'text',
    nullable: true,
  },
  analyzePrompt: {
    type: 'text',
    nullable: true,
  },
  reasonPrompt: {
    type: 'text',
    nullable: true,
  },
  input: {
    type: 'jsonb', // MESSAGE INPUT
  },
  output: {
    type: 'jsonb', // MESSAGE OUTPUT
  },
  additionalContext: {
    type: 'jsonb', // DATA FROM THE CONTEXT PARAM ON AN AGENT
    nullable: true,
  },
  runtimeContext: {
    type: 'jsonb', // THE EVALUATE RUNTIME CONTEXT FOR THE RUN
    nullable: true,
  },
  /**
   * Things you can evaluate
   */
  entityType: {
    type: 'text', // WORKFLOW, AGENT, TOOL, STEP, NETWORK
    nullable: true,
  },
  entity: {
    type: 'jsonb', // MINIMAL JSON DATA ABOUT WORKFLOW, AGENT, TOOL, STEP, NETWORK
    nullable: true,
  },
  entityId: {
    type: 'text',
    nullable: true,
  },
  source: {
    type: 'text',
  },
  resourceId: {
    type: 'text',
    nullable: true,
  },
  threadId: {
    type: 'text',
    nullable: true,
  },
  createdAt: {
    type: 'timestamp',
  },
  updatedAt: {
    type: 'timestamp',
  },
};

export const TABLE_SCHEMAS: Record<TABLE_NAMES, Record<string, StorageColumn>> = {
  [TABLE_WORKFLOW_SNAPSHOT]: {
    workflow_name: {
      type: 'text',
    },
    run_id: {
      type: 'text',
    },
    resourceId: { type: 'text', nullable: true },
    snapshot: {
      type: 'text',
    },
    createdAt: {
      type: 'timestamp',
    },
    updatedAt: {
      type: 'timestamp',
    },
  },
  [TABLE_SCORERS]: SCORERS_SCHEMA,
  [TABLE_EVALS]: {
    input: {
      type: 'text',
    },
    output: {
      type: 'text',
    },
    result: {
      type: 'jsonb',
    },
    agent_name: {
      type: 'text',
    },
    metric_name: {
      type: 'text',
    },
    instructions: {
      type: 'text',
    },
    test_info: {
      type: 'jsonb',
      nullable: true,
    },
    global_run_id: {
      type: 'text',
    },
    run_id: {
      type: 'text',
    },
    created_at: {
      type: 'timestamp',
    },
    createdAt: {
      type: 'timestamp',
      nullable: true,
    },
  },
  [TABLE_THREADS]: {
    id: { type: 'text', nullable: false, primaryKey: true },
    resourceId: { type: 'text', nullable: false },
    title: { type: 'text', nullable: false },
    metadata: { type: 'text', nullable: true },
    createdAt: { type: 'timestamp', nullable: false },
    updatedAt: { type: 'timestamp', nullable: false },
  },
  [TABLE_MESSAGES]: {
    id: { type: 'text', nullable: false, primaryKey: true },
    thread_id: { type: 'text', nullable: false },
    content: { type: 'text', nullable: false },
    role: { type: 'text', nullable: false },
    type: { type: 'text', nullable: false },
    createdAt: { type: 'timestamp', nullable: false },
    resourceId: { type: 'text', nullable: true },
  },
  [TABLE_TRACES]: {
    id: { type: 'text', nullable: false, primaryKey: true },
    parentSpanId: { type: 'text', nullable: true },
    name: { type: 'text', nullable: false },
    traceId: { type: 'text', nullable: false },
    scope: { type: 'text', nullable: false },
    kind: { type: 'integer', nullable: false },
    attributes: { type: 'jsonb', nullable: true },
    status: { type: 'jsonb', nullable: true },
    events: { type: 'jsonb', nullable: true },
    links: { type: 'jsonb', nullable: true },
    other: { type: 'text', nullable: true },
    startTime: { type: 'bigint', nullable: false },
    endTime: { type: 'bigint', nullable: false },
    createdAt: { type: 'timestamp', nullable: false },
  },
  [TABLE_RESOURCES]: {
    id: { type: 'text', nullable: false, primaryKey: true },
    workingMemory: { type: 'text', nullable: true },
    metadata: { type: 'jsonb', nullable: true },
    createdAt: { type: 'timestamp', nullable: false },
    updatedAt: { type: 'timestamp', nullable: false },
  },
};
