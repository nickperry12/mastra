import type { WorkflowRunState } from '../../../workflows';
import { TABLE_WORKFLOW_SNAPSHOT } from '../../constants';
import type { StorageWorkflowRun, WorkflowRun, WorkflowRuns } from '../../types';
import type { StoreOperations } from '../operations';
import { WorkflowsStorage } from './base';

export type InMemoryWorkflows = Map<string, StorageWorkflowRun>;

export class WorkflowsInMemory extends WorkflowsStorage {
  operations: StoreOperations;
  collection: InMemoryWorkflows;

  constructor({ collection, operations }: { collection: InMemoryWorkflows; operations: StoreOperations }) {
    super();
    this.collection = collection;
    this.operations = operations;
  }

  async persistWorkflowSnapshot({
    workflowName,
    runId,
    snapshot,
  }: {
    workflowName: string;
    runId: string;
    snapshot: WorkflowRunState;
  }) {
    const data = {
      workflow_name: workflowName,
      run_id: runId,
      snapshot,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    console.log('[persistWorkflowSnapshot] args:', { workflowName, runId, snapshot });
    console.log('[persistWorkflowSnapshot] data:', data);
    await this.operations.insert({
      tableName: TABLE_WORKFLOW_SNAPSHOT,
      record: data,
    });
  }

  async loadWorkflowSnapshot({
    workflowName,
    runId,
  }: {
    workflowName: string;
    runId: string;
  }): Promise<WorkflowRunState | null> {
    this.logger.debug('Loading workflow snapshot', { workflowName, runId });
    const d = await this.operations.load<{ snapshot: WorkflowRunState }>({
      tableName: TABLE_WORKFLOW_SNAPSHOT,
      keys: { workflow_name: workflowName, run_id: runId },
    });

    // Return a deep copy to prevent mutation
    return d ? JSON.parse(JSON.stringify(d.snapshot)) : null;
  }

  async getWorkflowRuns({
    workflowName,
    fromDate,
    toDate,
    limit,
    offset,
    resourceId,
  }: {
    workflowName?: string;
    fromDate?: Date;
    toDate?: Date;
    limit?: number;
    offset?: number;
    resourceId?: string;
  } = {}): Promise<WorkflowRuns> {
    console.log(`[getWorkflowRuns] called with`, { workflowName, fromDate, toDate, limit, offset, resourceId });
    let runs = Array.from(this.collection.values());
    console.log(`[getWorkflowRuns] initial runs:`, runs);

    if (workflowName) runs = runs.filter((run: any) => run.workflow_name === workflowName);
    if (fromDate && toDate) {
      runs = runs.filter(
        (run: any) =>
          new Date(run.createdAt).getTime() >= fromDate.getTime() &&
          new Date(run.createdAt).getTime() <= toDate.getTime(),
      );
    } else if (fromDate) {
      runs = runs.filter((run: any) => new Date(run.createdAt).getTime() >= fromDate.getTime());
    } else if (toDate) {
      runs = runs.filter((run: any) => new Date(run.createdAt).getTime() <= toDate.getTime());
    }
    if (resourceId) runs = runs.filter((run: any) => run.resourceId === resourceId);

    console.log(`[getWorkflowRuns] after filtering:`, runs);
    const total = runs.length;

    // Sort by createdAt
    runs.sort((a: any, b: any) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

    // Apply pagination
    if (limit !== undefined && offset !== undefined) {
      const start = offset;
      const end = start + limit;
      runs = runs.slice(start, end);
    }

    // Deserialize snapshot if it's a string
    const parsedRuns = runs.map((run: any) => ({
      ...run,
      snapshot: typeof run.snapshot === 'string' ? JSON.parse(run.snapshot) : JSON.parse(JSON.stringify(run.snapshot)),
      createdAt: new Date(run.createdAt),
      updatedAt: new Date(run.updatedAt),
      runId: run.run_id,
      workflowName: run.workflow_name,
    }));

    console.log(`[getWorkflowRuns] parsedRuns:`, parsedRuns);
    return { runs: parsedRuns as WorkflowRun[], total };
  }

  async getWorkflowRunById({
    runId,
    workflowName,
  }: {
    runId: string;
    workflowName?: string;
  }): Promise<WorkflowRun | null> {
    console.log(`[getWorkflowRunById] called for runId ${runId}, workflowName ${workflowName}`);
    let run = Array.from(this.collection.values()).find((r: any) => r.run_id === runId);

    if (run && workflowName && run.workflow_name !== workflowName) {
      run = undefined; // Not found if workflowName doesn't match
    }

    if (!run) return null;

    // Return a deep copy to prevent mutation
    const parsedRun = {
      ...run,
      snapshot: typeof run.snapshot === 'string' ? JSON.parse(run.snapshot) : JSON.parse(JSON.stringify(run.snapshot)),
      createdAt: new Date(run.createdAt),
      updatedAt: new Date(run.updatedAt),
      runId: run.run_id,
      workflowName: run.workflow_name,
    };

    console.log(`[getWorkflowRunById] parsedRun:`, parsedRun);
    return parsedRun as WorkflowRun;
  }
}
