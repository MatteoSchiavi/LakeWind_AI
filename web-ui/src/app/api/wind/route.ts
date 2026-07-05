/**
 * LakeWind API — reads wind forecasts from the DuckDB database.
 *
 * This route queries the LakeWind DuckDB file directly (read-only) and returns
 * JSON for the Next.js dashboard to consume.
 *
 * The DuckDB file lives at /app/data/lakewind.duckdb inside the LakeWind
 * Docker container. For development, we point to a local path.
 */
import { NextRequest, NextResponse } from 'next/server';
import duckdb from 'duckdb';

// Path to the DuckDB file — configurable via env var
const DB_PATH = process.env.LAKEWIND_DB_PATH || '/app/data/lakewind.duckdb';

function getDb(): duckdb.Database {
  // Open connection (Node duckdb doesn't support readonly option reliably)
  return new duckdb.Database(DB_PATH);
}

interface Prediction {
  point_id: string;
  generated_at: string;
  valid_time: string;
  model_version: string;
  wind_speed_kn: number | null;
  wind_dir_deg: number | null;
  wind_gust_kn: number | null;
  confidence_pct: number | null;
  expected_error_kn: number | null;
}

interface PointInfo {
  id: string;
  lat: number;
  lon: number;
  is_operational: boolean;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const horizon = searchParams.get('horizon') || '0'; // hours ahead
  const pointId = searchParams.get('point');

  try {
    const db = getDb();
    const horizonHours = parseInt(horizon, 10) || 0;

    // Target time = now + horizon hours
    const now = new Date();
    const targetTime = new Date(now.getTime() + horizonHours * 3600 * 1000);

    // Query: latest prediction for each operational point near target time
    let sql: string;
    let params: unknown[];

    if (pointId) {
      sql = `
        SELECT point_id, generated_at::TEXT as generated_at, valid_time::TEXT as valid_time,
               model_version, wind_speed_kn, wind_dir_deg, wind_gust_kn,
               confidence_pct, expected_error_kn
        FROM predictions
        WHERE point_id = ?
          AND ABS(EXTRACT(EPOCH FROM (valid_time - ?::TIMESTAMP))) < 3600
        ORDER BY generated_at DESC
        LIMIT 1
      `;
      params = [pointId, targetTime.toISOString()];
    } else {
      // Get the latest prediction for each operational point
      sql = `
        WITH ranked AS (
          SELECT point_id, generated_at::TEXT as generated_at, valid_time::TEXT as valid_time,
                 model_version, wind_speed_kn, wind_dir_deg, wind_gust_kn,
                 confidence_pct, expected_error_kn,
            ROW_NUMBER() OVER (PARTITION BY point_id ORDER BY generated_at DESC) as rn
          FROM predictions
          WHERE ABS(EXTRACT(EPOCH FROM (valid_time - ?::TIMESTAMP))) < 3600
        )
        SELECT point_id, generated_at, valid_time, model_version,
               wind_speed_kn, wind_dir_deg, wind_gust_kn, confidence_pct, expected_error_kn
        FROM ranked WHERE rn = 1
      `;
      params = [targetTime.toISOString()];
    }

    const predictions: Prediction[] = await new Promise((resolve, reject) => {
      db.all(sql, ...params, (err: Error | null, rows: unknown[]) => {
        if (err) reject(err);
        else resolve(rows as Prediction[]);
      });
    });

    db.close();

    return NextResponse.json({
      status: 'ok',
      target_time: targetTime.toISOString(),
      horizon_hours: horizonHours,
      predictions,
    });
  } catch (error) {
    console.error('LakeWind API error:', error);
    return NextResponse.json(
      {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        predictions: [],
      },
      { status: 500 }
    );
  }
}
