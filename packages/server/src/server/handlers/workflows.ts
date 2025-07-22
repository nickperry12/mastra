import { ReadableStream } from 'node:stream/web';
import type { RuntimeContext } from '@mastra/core/di';
import type { WorkflowRuns } from '@mastra/core/storage';
import type { Workflow, SerializedStepFlowEntry, WatchEvent, StepWithComponent } from '@mastra/core/workflows';
import { stringify } from 'superjson';
import zodToJsonSchema from 'zod-to-json-schema';
import { HTTPException } from '../http-exception';
import type { Context } from '../types';
import { handleError } from './error';

interface WorkflowContext extends Context {
  workflowId?: string;
  runId?: string;
}

function getSteps(steps: Record<string, StepWithComponent>, path?: string) {
  return Object.entries(steps).reduce<any>((acc, [key, step]) => {
    const fullKey = path ? `${path}.${key}` : key;
    acc[fullKey] = {
      id: step.id,
      description: step.description,
      inputSchema: step.inputSchema ? stringify(zodToJsonSchema(step.inputSchema)) : undefined,
      outputSchema: step.outputSchema ? stringify(zodToJsonSchema(step.outputSchema)) : undefined,
      resumeSchema: step.resumeSchema ? stringify(zodToJsonSchema(step.resumeSchema)) : undefined,
      suspendSchema: step.suspendSchema ? stringify(zodToJsonSchema(step.suspendSchema)) : undefined,
      isWorkflow: step.component === 'WORKFLOW',
    };

    if (step.component === 'WORKFLOW' && step.steps) {
      const nestedSteps = getSteps(step.steps, fullKey) || {};
      acc = { ...acc, ...nestedSteps };
    }

    return acc;
  }, {});
}

export async function getWorkflowsHandler({ mastra }: WorkflowContext) {
  try {
    const workflows = mastra.getWorkflows({ serialized: false });
    const _workflows = Object.entries(workflows).reduce<any>((acc, [key, workflow]) => {
      acc[key] = {
        name: workflow.name,
        description: workflow.description,
        steps: Object.entries(workflow.steps).reduce<any>((acc, [key, step]) => {
          acc[key] = {
            id: step.id,
            description: step.description,
            inputSchema: step.inputSchema ? stringify(zodToJsonSchema(step.inputSchema)) : undefined,
            outputSchema: step.outputSchema ? stringify(zodToJsonSchema(step.outputSchema)) : undefined,
            resumeSchema: step.resumeSchema ? stringify(zodToJsonSchema(step.resumeSchema)) : undefined,
            suspendSchema: step.suspendSchema ? stringify(zodToJsonSchema(step.suspendSchema)) : undefined,
          };
          return acc;
        }, {}),
        allSteps: getSteps(workflow.steps) || {},
        stepGraph: workflow.serializedStepGraph,
        inputSchema: workflow.inputSchema ? stringify(zodToJsonSchema(workflow.inputSchema)) : undefined,
        outputSchema: workflow.outputSchema ? stringify(zodToJsonSchema(workflow.outputSchema)) : undefined,
      };
      return acc;
    }, {});
    return _workflows;
  } catch (error) {
    return handleError(error, 'Error getting workflows');
  }
}

type SerializedStep = {
  id: string;
  description: string;
  inputSchema: string | undefined;
  outputSchema: string | undefined;
  resumeSchema: string | undefined;
  suspendSchema: string | undefined;
};

async function getWorkflowsFromSystem({ mastra, workflowId }: WorkflowContext) {
  const logger = mastra.getLogger();

  if (!workflowId) {
    throw new HTTPException(400, { message: 'Workflow ID is required' });
  }

  let workflow;

  try {
    workflow = mastra.getWorkflow(workflowId);
  } catch (error) {
    logger.debug('Error getting workflow, searching agents for workflow', error);
  }

  if (!workflow) {
    logger.debug('Workflow not found, searching agents for workflow', { workflowId });
    const agents = mastra.getAgents();

    if (Object.keys(agents || {}).length) {
      for (const [_, agent] of Object.entries(agents)) {
        try {
          const workflows = await agent.getWorkflows();

          if (workflows[workflowId]) {
            workflow = workflows[workflowId];
            break;
          }
          break;
        } catch (error) {
          logger.debug('Error getting workflow from agent', error);
        }
      }
    }
  }

  if (!workflow) {
    throw new HTTPException(404, { message: 'Workflow not found' });
  }

  return { workflow };
}

export async function getWorkflowByIdHandler({ mastra, workflowId }: WorkflowContext): Promise<{
  steps: Record<string, SerializedStep>;
  allSteps: Record<string, SerializedStep>;
  name: string | undefined;
  description: string | undefined;
  stepGraph: SerializedStepFlowEntry[];
  inputSchema: string | undefined;
  outputSchema: string | undefined;
}> {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    return {
      steps: Object.entries(workflow.steps).reduce<any>((acc, [key, step]) => {
        acc[key] = {
          id: step.id,
          description: step.description,
          inputSchema: step.inputSchema ? stringify(zodToJsonSchema(step.inputSchema)) : undefined,
          outputSchema: step.outputSchema ? stringify(zodToJsonSchema(step.outputSchema)) : undefined,
          resumeSchema: step.resumeSchema ? stringify(zodToJsonSchema(step.resumeSchema)) : undefined,
          suspendSchema: step.suspendSchema ? stringify(zodToJsonSchema(step.suspendSchema)) : undefined,
        };
        return acc;
      }, {}),
      allSteps: getSteps(workflow.steps) || {},
      name: workflow.name,
      description: workflow.description,
      stepGraph: workflow.serializedStepGraph,
      inputSchema: workflow.inputSchema ? stringify(zodToJsonSchema(workflow.inputSchema)) : undefined,
      outputSchema: workflow.outputSchema ? stringify(zodToJsonSchema(workflow.outputSchema)) : undefined,
    };
  } catch (error) {
    return handleError(error, 'Error getting workflow');
  }
}

export async function getWorkflowRunByIdHandler({
  mastra,
  workflowId,
  runId,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'>): Promise<ReturnType<Workflow['getWorkflowRunById']>> {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'Run ID is required' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.getWorkflowRunById(runId);

    if (!run) {
      throw new HTTPException(404, { message: 'Workflow run not found' });
    }

    return run;
  } catch (error) {
    return handleError(error, 'Error getting workflow run');
  }
}

export async function getWorkflowRunExecutionResultHandler({
  mastra,
  workflowId,
  runId,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'>): Promise<WatchEvent['payload']['workflowState']> {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'Run ID is required' });
    }

    const workflow = mastra.getWorkflow(workflowId);

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const executionResult = await workflow.getWorkflowRunExecutionResult(runId);

    if (!executionResult) {
      throw new HTTPException(404, { message: 'Workflow run execution result not found' });
    }

    return executionResult;
  } catch (error) {
    return handleError(error, 'Error getting workflow run execution result');
  }
}

export async function createWorkflowRunHandler({
  mastra,
  workflowId,
  runId: prevRunId,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'>) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.createRunAsync({ runId: prevRunId });

    return { runId: run.runId };
  } catch (error) {
    return handleError(error, 'Error creating workflow run');
  }
}

export async function startAsyncWorkflowHandler({
  mastra,
  runtimeContext,
  workflowId,
  runId,
  inputData,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'> & {
  inputData?: unknown;
  runtimeContext?: RuntimeContext;
}) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const _run = await workflow.createRunAsync({ runId });
    const result = await _run.start({
      inputData,
      runtimeContext,
    });
    return result;
  } catch (error) {
    return handleError(error, 'Error starting async workflow');
  }
}

export async function startWorkflowRunHandler({
  mastra,
  runtimeContext,
  workflowId,
  runId,
  inputData,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'> & {
  inputData?: unknown;
  runtimeContext?: RuntimeContext;
}) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to start run' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.getWorkflowRunById(runId);

    if (!run) {
      throw new HTTPException(404, { message: 'Workflow run not found' });
    }

    const _run = await workflow.createRunAsync({ runId });
    void _run.start({
      inputData,
      runtimeContext,
    });

    return { message: 'Workflow run started' };
  } catch (e) {
    return handleError(e, 'Error starting workflow run');
  }
}

export async function watchWorkflowHandler({
  mastra,
  workflowId,
  runId,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'>): Promise<ReadableStream<string>> {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to watch workflow' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.getWorkflowRunById(runId);

    if (!run) {
      throw new HTTPException(404, { message: 'Workflow run not found' });
    }

    const _run = await workflow.createRunAsync({ runId });
    let unwatch: () => void;
    let asyncRef: NodeJS.Immediate | null = null;
    const stream = new ReadableStream<string>({
      start(controller) {
        unwatch = _run.watch(({ type, payload, eventTimestamp }) => {
          controller.enqueue(JSON.stringify({ type, payload, eventTimestamp, runId }));

          if (asyncRef) {
            clearImmediate(asyncRef);
            asyncRef = null;
          }

          // a run is finished if the status is not running
          asyncRef = setImmediate(async () => {
            const runDone = payload.workflowState.status !== 'running';
            if (runDone) {
              controller.close();
              unwatch?.();
            }
          });
        });
      },
      cancel() {
        unwatch?.();
      },
    });

    return stream;
  } catch (error) {
    return handleError(error, 'Error watching workflow');
  }
}

export async function streamWorkflowHandler({
  mastra,
  runtimeContext,
  workflowId,
  runId,
  inputData,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'> & {
  inputData?: unknown;
  runtimeContext?: RuntimeContext;
}) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to resume workflow' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.createRunAsync({ runId });
    const result = run.stream({
      inputData,
      runtimeContext,
    });
    return result;
  } catch (error) {
    return handleError(error, 'Error executing workflow');
  }
}

export async function streamVNextWorkflowHandler({
  mastra,
  runtimeContext,
  workflowId,
  runId,
  inputData,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'> & {
  inputData?: unknown;
  runtimeContext?: RuntimeContext;
}) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to stream workflow' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.createRunAsync({ runId });
    const result = run.streamVNext({
      inputData,
      runtimeContext,
    });
    return result;
  } catch (error) {
    return handleError(error, 'Error streaming workflow');
  }
}

export async function resumeAsyncWorkflowHandler({
  mastra,
  workflowId,
  runId,
  body,
  runtimeContext,
}: WorkflowContext & {
  body: { step: string | string[]; resumeData?: unknown };
  runtimeContext?: RuntimeContext;
}) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to resume workflow' });
    }

    if (!body.step) {
      throw new HTTPException(400, { message: 'step required to resume workflow' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.getWorkflowRunById(runId);

    if (!run) {
      throw new HTTPException(404, { message: 'Workflow run not found' });
    }

    const _run = await workflow.createRunAsync({ runId });
    const result = await _run.resume({
      step: body.step,
      resumeData: body.resumeData,
      runtimeContext,
    });

    return result;
  } catch (error) {
    return handleError(error, 'Error resuming workflow step');
  }
}

export async function resumeWorkflowHandler({
  mastra,
  workflowId,
  runId,
  body,
  runtimeContext,
}: WorkflowContext & {
  body: { step: string | string[]; resumeData?: unknown };
  runtimeContext?: RuntimeContext;
}) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to resume workflow' });
    }

    if (!body.step) {
      throw new HTTPException(400, { message: 'step required to resume workflow' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.getWorkflowRunById(runId);

    if (!run) {
      throw new HTTPException(404, { message: 'Workflow run not found' });
    }

    const _run = await workflow.createRunAsync({ runId });

    void _run.resume({
      step: body.step,
      resumeData: body.resumeData,
      runtimeContext,
    });

    return { message: 'Workflow run resumed' };
  } catch (error) {
    return handleError(error, 'Error resuming workflow');
  }
}

export async function getWorkflowRunsHandler({
  mastra,
  workflowId,
  fromDate,
  toDate,
  limit,
  offset,
  resourceId,
}: WorkflowContext & {
  fromDate?: Date;
  toDate?: Date;
  limit?: number;
  offset?: number;
  resourceId?: string;
}): Promise<WorkflowRuns> {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const workflowRuns = (await workflow.getWorkflowRuns({ fromDate, toDate, limit, offset, resourceId })) || {
      runs: [],
      total: 0,
    };
    return workflowRuns;
  } catch (error) {
    return handleError(error, 'Error getting workflow runs');
  }
}

export async function cancelWorkflowRunHandler({
  mastra,
  workflowId,
  runId,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'>) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to cancel workflow run' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.getWorkflowRunById(runId);

    if (!run) {
      throw new HTTPException(404, { message: 'Workflow run not found' });
    }

    const _run = await workflow.createRunAsync({ runId });

    await _run.cancel();

    return { message: 'Workflow run cancelled' };
  } catch (error) {
    return handleError(error, 'Error canceling workflow run');
  }
}

export async function sendWorkflowRunEventHandler({
  mastra,
  workflowId,
  runId,
  event,
  data,
}: Pick<WorkflowContext, 'mastra' | 'workflowId' | 'runId'> & {
  event: string;
  data: unknown;
}) {
  try {
    if (!workflowId) {
      throw new HTTPException(400, { message: 'Workflow ID is required' });
    }

    if (!runId) {
      throw new HTTPException(400, { message: 'runId required to send workflow run event' });
    }

    const { workflow } = await getWorkflowsFromSystem({ mastra, workflowId });

    if (!workflow) {
      throw new HTTPException(404, { message: 'Workflow not found' });
    }

    const run = await workflow.getWorkflowRunById(runId);

    if (!run) {
      throw new HTTPException(404, { message: 'Workflow run not found' });
    }

    const _run = await workflow.createRunAsync({ runId });

    await _run.sendEvent(event, data);

    return { message: 'Workflow run event sent' };
  } catch (error) {
    return handleError(error, 'Error sending workflow run event');
  }
}
