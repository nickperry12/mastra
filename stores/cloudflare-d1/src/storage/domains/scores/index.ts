import { ErrorDomain, ErrorCategory, MastraError } from '@mastra/core/error';
import type { ScoreRowData } from '@mastra/core/scores';
import { ScoresStorage, TABLE_SCORERS } from '@mastra/core/storage';
import type { StoragePagination, PaginationInfo } from '@mastra/core/storage';
import type Cloudflare from 'cloudflare';
import { createSqlBuilder } from '../../sql-builder';
import type { StoreOperationsD1 } from '../operations';

export type D1QueryResult = Awaited<ReturnType<Cloudflare['d1']['database']['query']>>['result'];

export interface D1Client {
  query(args: { sql: string; params: string[] }): Promise<{ result: D1QueryResult }>;
}

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

export class ScoresStorageD1 extends ScoresStorage {
  private operations: StoreOperationsD1;

  constructor({ operations }: { operations: StoreOperationsD1 }) {
    super();
    this.operations = operations;
  }

  async getScoreById({ id }: { id: string }): Promise<ScoreRowData | null> {
    try {
      const fullTableName = this.operations.getTableName(TABLE_SCORERS);
      const query = createSqlBuilder().select('*').from(fullTableName).where('id = ?', id);
      const { sql, params } = query.build();

      const result = await this.operations.executeQuery({ sql, params, first: true });

      if (!result) {
        return null;
      }

      return transformScoreRow(result);
    } catch (error) {
      throw new MastraError(
        {
          id: 'CLOUDFLARE_D1_STORE_SCORES_GET_SCORE_BY_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  async saveScore(score: Omit<ScoreRowData, 'createdAt' | 'updatedAt'>): Promise<{ score: ScoreRowData }> {
    try {
      const fullTableName = this.operations.getTableName(TABLE_SCORERS);
      const { input, ...rest } = score;

      // Serialize all object values to JSON strings
      const serializedRecord: Record<string, any> = {};
      for (const [key, value] of Object.entries(rest)) {
        if (value !== null && value !== undefined) {
          if (typeof value === 'object') {
            serializedRecord[key] = JSON.stringify(value);
          } else {
            serializedRecord[key] = value;
          }
        } else {
          serializedRecord[key] = null;
        }
      }

      serializedRecord.input = JSON.stringify(input);
      serializedRecord.createdAt = new Date().toISOString();
      serializedRecord.updatedAt = new Date().toISOString();

      const columns = Object.keys(serializedRecord);
      const values = Object.values(serializedRecord);

      const query = createSqlBuilder().insert(fullTableName, columns, values);
      const { sql, params } = query.build();

      await this.operations.executeQuery({ sql, params });

      const scoreFromDb = await this.getScoreById({ id: score.id });
      return { score: scoreFromDb! };
    } catch (error) {
      throw new MastraError(
        {
          id: 'CLOUDFLARE_D1_STORE_SCORES_SAVE_SCORE_FAILED',
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
      const fullTableName = this.operations.getTableName(TABLE_SCORERS);

      // Get total count
      const countQuery = createSqlBuilder().count().from(fullTableName).where('scorerId = ?', scorerId);
      const countResult = await this.operations.executeQuery(countQuery.build());
      const total = Array.isArray(countResult) ? Number(countResult?.[0]?.count ?? 0) : Number(countResult?.count ?? 0);

      if (total === 0) {
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

      // Get paginated results
      const selectQuery = createSqlBuilder()
        .select('*')
        .from(fullTableName)
        .where('scorerId = ?', scorerId)
        .limit(pagination.perPage)
        .offset(pagination.page * pagination.perPage);

      const { sql, params } = selectQuery.build();
      const results = await this.operations.executeQuery({ sql, params });

      const scores = Array.isArray(results) ? results.map(transformScoreRow) : [];

      return {
        pagination: {
          total,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: total > (pagination.page + 1) * pagination.perPage,
        },
        scores,
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'CLOUDFLARE_D1_STORE_SCORES_GET_SCORES_BY_SCORER_ID_FAILED',
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
      const fullTableName = this.operations.getTableName(TABLE_SCORERS);

      // Get total count
      const countQuery = createSqlBuilder().count().from(fullTableName).where('runId = ?', runId);
      const countResult = await this.operations.executeQuery(countQuery.build());
      const total = Array.isArray(countResult) ? Number(countResult?.[0]?.count ?? 0) : Number(countResult?.count ?? 0);

      if (total === 0) {
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

      // Get paginated results
      const selectQuery = createSqlBuilder()
        .select('*')
        .from(fullTableName)
        .where('runId = ?', runId)
        .limit(pagination.perPage)
        .offset(pagination.page * pagination.perPage);

      const { sql, params } = selectQuery.build();
      const results = await this.operations.executeQuery({ sql, params });

      const scores = Array.isArray(results) ? results.map(transformScoreRow) : [];

      return {
        pagination: {
          total,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: total > (pagination.page + 1) * pagination.perPage,
        },
        scores,
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'CLOUDFLARE_D1_STORE_SCORES_GET_SCORES_BY_RUN_ID_FAILED',
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
      const fullTableName = this.operations.getTableName(TABLE_SCORERS);

      // Get total count
      const countQuery = createSqlBuilder()
        .count()
        .from(fullTableName)
        .where('entityId = ?', entityId)
        .andWhere('entityType = ?', entityType);
      const countResult = await this.operations.executeQuery(countQuery.build());
      const total = Array.isArray(countResult) ? Number(countResult?.[0]?.count ?? 0) : Number(countResult?.count ?? 0);

      if (total === 0) {
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

      // Get paginated results
      const selectQuery = createSqlBuilder()
        .select('*')
        .from(fullTableName)
        .where('entityId = ?', entityId)
        .andWhere('entityType = ?', entityType)
        .limit(pagination.perPage)
        .offset(pagination.page * pagination.perPage);

      const { sql, params } = selectQuery.build();
      const results = await this.operations.executeQuery({ sql, params });

      const scores = Array.isArray(results) ? results.map(transformScoreRow) : [];

      return {
        pagination: {
          total,
          page: pagination.page,
          perPage: pagination.perPage,
          hasMore: total > (pagination.page + 1) * pagination.perPage,
        },
        scores,
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'CLOUDFLARE_D1_STORE_SCORES_GET_SCORES_BY_ENTITY_ID_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }
}
