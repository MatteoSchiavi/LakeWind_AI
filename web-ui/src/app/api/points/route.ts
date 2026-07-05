import { NextResponse } from 'next/server';

interface PointInfo {
  id: string;
  lat: number;
  lon: number;
  is_operational: boolean;
  sector: string;
}

// Virtual points from settings.yaml (V3: 15 operational + 4 auxiliary)
const VIRTUAL_POINTS: PointInfo[] = [
  // North sector (Dongo-Gravedona-Domaso)
  { id: 'dongo_shore', lat: 46.124, lon: 9.280, is_operational: true, sector: 'North' },
  { id: 'dongo_offshore', lat: 46.130, lon: 9.295, is_operational: true, sector: 'North' },
  { id: 'gravedona_shore', lat: 46.147, lon: 9.307, is_operational: true, sector: 'North' },
  { id: 'gravedona_offshore', lat: 46.145, lon: 9.320, is_operational: true, sector: 'North' },
  { id: 'domaso_offshore', lat: 46.151, lon: 9.332, is_operational: true, sector: 'North' },
  // Mid-lake sector (Musso-Piona-Dervio)
  { id: 'musso_shore', lat: 46.116, lon: 9.290, is_operational: true, sector: 'Mid-lake' },
  { id: 'mid_channel_north', lat: 46.110, lon: 9.300, is_operational: true, sector: 'Mid-lake' },
  { id: 'mid_channel', lat: 46.100, lon: 9.304, is_operational: true, sector: 'Mid-lake' },
  { id: 'piona_entrance', lat: 46.114, lon: 9.316, is_operational: true, sector: 'Mid-lake' },
  { id: 'dervio_shore', lat: 46.077, lon: 9.307, is_operational: true, sector: 'Mid-lake' },
  { id: 'dervio_offshore', lat: 46.075, lon: 9.315, is_operational: true, sector: 'Mid-lake' },
  // South sector (Bellano-Lecno)
  { id: 'bellano_offshore', lat: 46.051, lon: 9.304, is_operational: true, sector: 'South' },
  { id: 'bellano_shore', lat: 46.050, lon: 9.294, is_operational: true, sector: 'South' },
  { id: 'lecco_north', lat: 46.020, lon: 9.290, is_operational: true, sector: 'South' },
  { id: 'valmadrera_south', lat: 46.010, lon: 9.330, is_operational: true, sector: 'South' },
];

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    points: VIRTUAL_POINTS,
    operational_count: VIRTUAL_POINTS.filter((p) => p.is_operational).length,
  });
}
