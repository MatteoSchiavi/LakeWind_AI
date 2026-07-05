import { NextRequest, NextResponse } from 'next/server';
import duckdb from "duckdb";

const DB_PATH = process.env.LAKEWIND_DB_PATH || '/app/data/lakewind.duckdb';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const pointId = searchParams.get('point') || 'mid_channel';
  const hours = parseInt(searchParams.get('hours') || '24', 10);

  try {
    const db = new duckdb.Database(DB_PATH);

    const rows: Array<{
      point_id: string;
      valid_time: string;
      wind_speed_kn: number | null;
      wind_dir_deg: number | null;
      wind_gust_kn: number | null;
      confidence_pct: number | null;
      expected_error_kn: number | null;
    }> = await new Promise((resolve, reject) => {
      db.all(
        `SELECT point_id, valid_time::TEXT as valid_time, wind_speed_kn, wind_dir_deg,
                wind_gust_kn, confidence_pct, expected_error_kn
         FROM predictions
         WHERE point_id = ?
           AND valid_time >= NOW() - INTERVAL '${hours + 1} hours'
           AND valid_time <= NOW() + INTERVAL '${hours} hours'
         ORDER BY valid_time ASC`,
        pointId,
        (err: Error | null, rows: unknown[]) => {
          if (err) reject(err);
          else resolve(rows as typeof rows);
        }
      );
    });

    db.close();

    return NextResponse.json({
      status: 'ok',
      point_id: pointId,
      hours,
      data: rows.map((r) => ({
        ...r,
        time: new Date(r.valid_time).getTime(),
      })),
    });
  } catch (error) {
    console.error('Trend API error:', error);
    return NextResponse.json(
      { status: 'error', error: error instanceof Error ? error.message : 'Unknown error', data: [] },
      { status: 500 }
    );
  }
}
