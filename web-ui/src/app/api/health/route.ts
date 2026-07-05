import { NextResponse } from 'next/server';
import duckdb from "duckdb";

const DB_PATH = process.env.LAKEWIND_DB_PATH || '/app/data/lakewind.duckdb';

export async function GET() {
  try {
    const db = new duckdb.Database(DB_PATH);

    const health: Array<{
      source: string;
      ok: boolean;
      latency_ms: number;
      checked_at: string;
      error_msg: string | null;
    }> = await new Promise((resolve, reject) => {
      db.all(
        `SELECT s.source, s.ok, s.latency_ms, s.checked_at::TEXT as checked_at, s.error_msg
         FROM source_health s
         JOIN (
           SELECT source, MAX(checked_at) AS m FROM source_health GROUP BY source
         ) m ON s.source = m.source AND s.checked_at = m.m
         ORDER BY s.source`,
        (err: Error | null, rows: unknown[]) => {
          if (err) reject(err);
          else resolve(rows as typeof health);
        }
      );
    });

    // Get latest prediction timestamp
    const lastPred: Array<{ generated_at: string }> = await new Promise((resolve, reject) => {
      db.all(
        'SELECT MAX(generated_at)::TEXT as generated_at FROM predictions',
        (err: Error | null, rows: unknown[]) => {
          if (err) reject(err);
          else resolve(rows as Array<{ generated_at: string }>);
        }
      );
    });

    // Count rows in key tables (cast COUNT to avoid BigInt)
    const counts: Record<string, number> = {};
    for (const table of ['forecast_runs', 'observations', 'predictions', 'model_registry']) {
      const result: Array<{ count: number }> = await new Promise((resolve, reject) => {
        db.all(
          `SELECT CAST(COUNT(*) AS INTEGER) as count FROM ${table}`,
          (err: Error | null, rows: unknown[]) => {
            if (err) reject(err);
            else resolve(rows as Array<{ count: number }>);
          }
        );
      });
      counts[table] = result[0]?.count || 0;
    }

    db.close();

    return NextResponse.json({
      status: 'ok',
      sources: health,
      last_prediction: lastPred[0]?.generated_at || null,
      table_counts: counts,
    });
  } catch (error) {
    console.error('Health API error:', error);
    return NextResponse.json(
      {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        sources: [],
      },
      { status: 500 }
    );
  }
}
