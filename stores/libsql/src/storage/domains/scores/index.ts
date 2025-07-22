import type { Client, InValue } from '@libsql/client';
import { ErrorCategory, ErrorDomain, MastraError } from '@mastra/core/error';
import type { ScoreRowData } from '@mastra/core/scores';
import { TABLE_SCORERS, ScoresStorage } from '@mastra/core/storage';
import type { PaginationInfo, StoragePagination } from '@mastra/core/storage';
import type { StoreOperationsLibSQL } from '../operations';

export class ScoresLibSQL extends ScoresStorage {
  private operations: StoreOperationsLibSQL;
  private client: Client;
  constructor({ client, operations }: { client: Client; operations: StoreOperationsLibSQL }) {
    super();
    this.operations = operations;
    this.client = client;
  }

  async getScoresByRunId({
    runId,
    pagination,
  }: {
    runId: string;
    pagination: StoragePagination;
  }): Promise<{ pagination: PaginationInfo; scores: ScoreRowData[] }> {
    try {
      const result = await this.client.execute({
        sql: `SELECT * FROM ${TABLE_SCORERS} WHERE runId = ? ORDER BY createdAt DESC LIMIT ? OFFSET ?`,
        args: [runId, pagination.perPage + 1, pagination.page * pagination.perPage],
      });
      return {
        scores: result.rows?.slice(0, pagination.perPage).map(row => this.transformScoreRow(row)) ?? [],
        pagination: {
          total: result.rows?.length ?? 0,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: result.rows?.length > pagination.perPage,
        },
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'LIBSQL_STORE_GET_SCORES_BY_RUN_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  async getScoresByScorerId({
    scorerId,
    entityId,
    entityType,
    pagination,
  }: {
    scorerId: string;
    entityId?: string;
    entityType?: string;
    pagination: StoragePagination;
  }): Promise<{ pagination: PaginationInfo; scores: ScoreRowData[] }> {
    try {
      const conditions: string[] = [];
      const queryParams: InValue[] = [];

      if (scorerId) {
        conditions.push(`scorerId = ?`);
        queryParams.push(scorerId);
      }

      if (entityId) {
        conditions.push(`entityId = ?`);
        queryParams.push(entityId);
      }

      if (entityType) {
        conditions.push(`entityType = ?`);
        queryParams.push(entityType);
      }

      const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

      const result = await this.client.execute({
        sql: `SELECT * FROM ${TABLE_SCORERS} ${whereClause} ORDER BY createdAt DESC LIMIT ? OFFSET ?`,
        args: [...queryParams, pagination.perPage + 1, pagination.page * pagination.perPage],
      });

      return {
        scores: result.rows?.slice(0, pagination.perPage).map(row => this.transformScoreRow(row)) ?? [],
        pagination: {
          total: result.rows?.length ?? 0,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: result.rows?.length > pagination.perPage,
        },
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'LIBSQL_STORE_GET_SCORES_BY_SCORER_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  private transformScoreRow(row: Record<string, any>): ScoreRowData {
    const scorerValue = JSON.parse(row.scorer ?? '{}');
    const inputValue = JSON.parse(row.input ?? '{}');
    const outputValue = JSON.parse(row.output ?? '{}');
    const additionalLLMContextValue = row.additionalLLMContext ? JSON.parse(row.additionalLLMContext) : null;
    const runtimeContextValue = row.runtimeContext ? JSON.parse(row.runtimeContext) : null;
    const metadataValue = row.metadata ? JSON.parse(row.metadata) : null;
    const entityValue = row.entity ? JSON.parse(row.entity) : null;
    const extractStepResultValue = row.extractStepResult ? JSON.parse(row.extractStepResult) : null;
    const analyzeStepResultValue = row.analyzeStepResult ? JSON.parse(row.analyzeStepResult) : null;

    return {
      id: row.id,
      traceId: row.traceId,
      runId: row.runId,
      scorer: scorerValue,
      score: row.score,
      reason: row.reason,
      extractStepResult: extractStepResultValue,
      analyzeStepResult: analyzeStepResultValue,
      analyzePrompt: row.analyzePrompt,
      extractPrompt: row.extractPrompt,
      metadata: metadataValue,
      input: inputValue,
      output: outputValue,
      additionalContext: additionalLLMContextValue,
      runtimeContext: runtimeContextValue,
      entityType: row.entityType,
      entity: entityValue,
      entityId: row.entityId,
      scorerId: row.scorerId,
      source: row.source,
      resourceId: row.resourceId,
      threadId: row.threadId,
      createdAt: row.createdAt,
      updatedAt: row.updatedAt,
    };
  }

  async getScoreById({ id }: { id: string }): Promise<ScoreRowData | null> {
    const result = await this.client.execute({
      sql: `SELECT * FROM ${TABLE_SCORERS} WHERE id = ?`,
      args: [id],
    });
    return result.rows?.[0] ? this.transformScoreRow(result.rows[0]) : null;
  }

  async saveScore(score: Omit<ScoreRowData, 'id' | 'createdAt' | 'updatedAt'>): Promise<{ score: ScoreRowData }> {
    try {
      const id = crypto.randomUUID();

      await this.operations.insert({
        tableName: TABLE_SCORERS,
        record: {
          id,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          ...score,
        },
      });

      const scoreFromDb = await this.getScoreById({ id });
      return { score: scoreFromDb! };
    } catch (error) {
      throw new MastraError(
        {
          id: 'LIBSQL_STORE_SAVE_SCORE_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  async getScoresByEntityId({
    entityId,
    entityType,
    pagination,
  }: {
    pagination: StoragePagination;
    entityId: string;
    entityType: string;
  }): Promise<{ pagination: PaginationInfo; scores: ScoreRowData[] }> {
    try {
      const result = await this.client.execute({
        sql: `SELECT * FROM ${TABLE_SCORERS} WHERE entityId = ? AND entityType = ? ORDER BY createdAt DESC LIMIT ? OFFSET ?`,
        args: [entityId, entityType, pagination.perPage + 1, pagination.page * pagination.perPage],
      });
      return {
        scores: result.rows?.slice(0, pagination.perPage).map(row => this.transformScoreRow(row)) ?? [],
        pagination: {
          total: result.rows?.length ?? 0,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: result.rows?.length > pagination.perPage,
        },
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'LIBSQL_STORE_GET_SCORES_BY_ENTITY_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }
}
