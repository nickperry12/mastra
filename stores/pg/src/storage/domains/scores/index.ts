import type { PaginationInfo, StoragePagination } from '@mastra/core';
import { ErrorCategory, ErrorDomain, MastraError } from '@mastra/core/error';
import type { ScoreRowData } from '@mastra/core/scores';
import { ScoresStorage, TABLE_SCORERS } from '@mastra/core/storage';
import type { IDatabase } from 'pg-promise';
import type { StoreOperationsPG } from '../operations';

function transformScoreRow(row: Record<string, any>): ScoreRowData {
  let input = undefined;

  if (row.input) {
    try {
      input = JSON.parse(row.input);
    } catch {
      input = row.input;
    }
  }
  return {
    ...row,
    input,
    createdAt: row.createdAtZ || row.createdAt,
    updatedAt: row.updatedAtZ || row.updatedAt,
  } as ScoreRowData;
}

export class ScoresPG extends ScoresStorage {
  public client: IDatabase<{}>;
  private operations: StoreOperationsPG;

  constructor({ client, operations }: { client: IDatabase<{}>; operations: StoreOperationsPG }) {
    super();
    this.client = client;
    this.operations = operations;
  }

  async getScoreById({ id }: { id: string }): Promise<ScoreRowData | null> {
    try {
      const result = await this.client.oneOrNone<ScoreRowData>(`SELECT * FROM ${TABLE_SCORERS} WHERE id = $1`, [id]);

      return transformScoreRow(result!);
    } catch (error) {
      throw new MastraError(
        {
          id: 'MASTRA_STORAGE_PG_STORE_GET_SCORE_BY_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  async getScoresByScorerId({
    scorerId,
    pagination,
  }: {
    scorerId: string;
    pagination: StoragePagination;
  }): Promise<{ pagination: PaginationInfo; scores: ScoreRowData[] }> {
    try {
      const total = await this.client.oneOrNone<{ count: string }>(
        `SELECT COUNT(*) FROM ${TABLE_SCORERS} WHERE "scorerId" = $1`,
        [scorerId],
      );
      if (total?.count === '0' || !total?.count) {
        return {
          pagination: {
            total: 0,
            page: pagination.page,
            perPage: pagination.perPage,
            hasMore: false,
          },
          scores: [],
        };
      }

      const result = await this.client.manyOrNone<ScoreRowData>(
        `SELECT * FROM ${TABLE_SCORERS} WHERE "scorerId" = $1 LIMIT $2 OFFSET $3`,
        [scorerId, pagination.perPage, pagination.page * pagination.perPage],
      );
      return {
        pagination: {
          total: Number(total?.count) || 0,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: Number(total?.count) > (pagination.page + 1) * pagination.perPage,
        },
        scores: result,
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'MASTRA_STORAGE_PG_STORE_GET_SCORES_BY_SCORER_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  async saveScore(score: Omit<ScoreRowData, 'createdAt' | 'updatedAt'>): Promise<{ score: ScoreRowData }> {
    try {
      const { input, ...rest } = score;
      await this.operations.insert({
        tableName: TABLE_SCORERS,
        record: {
          ...rest,
          input: JSON.stringify(input),
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      });

      const scoreFromDb = await this.getScoreById({ id: score.id });
      return { score: scoreFromDb! };
    } catch (error) {
      throw new MastraError(
        {
          id: 'MASTRA_STORAGE_PG_STORE_SAVE_SCORE_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  async getScoresByRunId({
    runId,
    pagination,
  }: {
    runId: string;
    pagination: StoragePagination;
  }): Promise<{ pagination: PaginationInfo; scores: ScoreRowData[] }> {
    try {
      const total = await this.client.oneOrNone<{ count: string }>(
        `SELECT COUNT(*) FROM ${TABLE_SCORERS} WHERE "runId" = $1`,
        [runId],
      );
      console.log(`total: ${total?.count}`);
      console.log(`typeof total: ${typeof total?.count}`);
      if (total?.count === '0' || !total?.count) {
        return {
          pagination: {
            total: 0,
            page: pagination.page,
            perPage: pagination.perPage,
            hasMore: false,
          },
          scores: [],
        };
      }

      const result = await this.client.manyOrNone<ScoreRowData>(
        `SELECT * FROM ${TABLE_SCORERS} WHERE "runId" = $1 LIMIT $2 OFFSET $3`,
        [runId, pagination.perPage, pagination.page * pagination.perPage],
      );
      return {
        pagination: {
          total: Number(total?.count) || 0,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: Number(total?.count) > (pagination.page + 1) * pagination.perPage,
        },
        scores: result,
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'MASTRA_STORAGE_PG_STORE_GET_SCORES_BY_RUN_ID_FAILED',
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
      const total = await this.client.oneOrNone<{ count: string }>(
        `SELECT COUNT(*) FROM ${TABLE_SCORERS} WHERE "entityId" = $1 AND "entityType" = $2`,
        [entityId, entityType],
      );

      if (total?.count === '0' || !total?.count) {
        return {
          pagination: {
            total: 0,
            page: pagination.page,
            perPage: pagination.perPage,
            hasMore: false,
          },
          scores: [],
        };
      }

      const result = await this.client.manyOrNone<ScoreRowData>(
        `SELECT * FROM ${TABLE_SCORERS} WHERE "entityId" = $1 AND "entityType" = $2 LIMIT $3 OFFSET $4`,
        [entityId, entityType, pagination.perPage, pagination.page * pagination.perPage],
      );
      return {
        pagination: {
          total: Number(total?.count) || 0,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: Number(total?.count) > (pagination.page + 1) * pagination.perPage,
        },
        scores: result,
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'MASTRA_STORAGE_PG_STORE_GET_SCORES_BY_ENTITY_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }
}
