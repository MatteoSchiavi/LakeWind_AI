"""Physics and convention tests: wind vectors, weather codes, solar, regime."""
from __future__ import annotations

from datetime import datetime

import pytest

from lakewind.utils.wind import WindVector, bias_correct, circular_direction_error_deg
from lakewind.utils.weather import (
    decode_weather_code,
    is_foggy,
    is_rainy,
    is_snowy,
    is_stormy,
    sailing_weather_warning,
)


class TestWindVector:
    def test_north_wind_blows_south(self):
        u, v = WindVector(10.0, 0.0).to_uv()
        assert u == pytest.approx(0.0, abs=1e-9)
        assert v == pytest.approx(-10.0, abs=1e-9)

    def test_east_wind_blows_west(self):
        u, v = WindVector(10.0, 90.0).to_uv()
        assert u == pytest.approx(-10.0, abs=1e-9)
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_roundtrip(self):
        for deg in range(0, 360, 15):
            w = WindVector(7.3, float(deg))
            u, v = w.to_uv()
            w2 = WindVector.from_uv(u, v)
            assert w2.speed_kn == pytest.approx(w.speed_kn, abs=1e-9)
            diff = circular_direction_error_deg(w2.direction_deg, w.direction_deg)
            assert diff < 1e-6

    def test_calm_direction_safe(self):
        w = WindVector.from_uv(0.0, 0.0)
        assert w.speed_kn == 0.0

    def test_bias_correct(self):
        out = bias_correct(1.0, 2.0, 0.5, -0.5)
        assert out.speed_kn == pytest.approx(WindVector.from_uv(1.5, 1.5).speed_kn)


class TestWeatherCodes:
    def test_decode(self):
        desc, icon = decode_weather_code(0, "en")
        assert desc == "Clear sky"
        desc_it, _ = decode_weather_code(61, "it")
        assert desc_it == "Pioggia leggera"

    def test_unknown_and_none(self):
        assert decode_weather_code(None) == ("Unknown", "❓")
        assert decode_weather_code(1234) == ("Unknown", "❓")

    def test_categories(self):
        assert is_rainy(63) and not is_rainy(3)
        assert is_snowy(73) and not is_snowy(61)
        assert is_stormy(95) and not is_stormy(80)
        assert is_foggy(45) and not is_foggy(2)

    def test_sailing_warnings(self):
        assert sailing_weather_warning(95, 10.0) is not None
        assert sailing_weather_warning(0, 10.0) is None
        # fog via visibility
        assert sailing_weather_warning(0, 5.0, visibility_m=500) is not None
        # heavy rain + strong wind
        assert sailing_weather_warning(65, 20.0) is not None


class TestSolar:
    def test_naive_treated_as_utc(self):
        """V6.6 REGRESSION GUARD: naive datetimes are UTC. At 12:00 UTC in July
        the sun must be high over Lake Como (~62° elevation). Under the old
        local-time storage the same sample computed ~10° lower."""
        from lakewind.utils.solar import solar_state_at

        s = solar_state_at(46.10, 9.304, datetime(2026, 7, 15, 12, 0))
        assert s.elevation_deg > 50.0, f"got {s.elevation_deg}"
        assert s.is_daytime

    def test_night(self):
        from lakewind.utils.solar import solar_state_at

        s = solar_state_at(46.10, 9.304, datetime(2026, 7, 15, 0, 0))
        assert not s.is_daytime
        assert s.elevation_deg < 0


class TestRegime:
    def test_priorities(self):
        from lakewind.ml.regime import classify_regime

        # foehn_strong wins over everything
        fv = {"foehn_strong": True, "foehn_likely": True,
              "fc_icon_eu_cape": 2000, "fc_icon_eu_speed": 20}
        res = classify_regime(datetime(2026, 7, 15, 12, 0), fv)
        assert res.regime == "foehn"

        # storm: high cape + strong wind, no foehn
        fv = {"foehn_likely": False, "fc_icon_eu_cape": 2000, "fc_icon_eu_speed": 20}
        res = classify_regime(datetime(2026, 7, 15, 12, 0), fv)
        assert res.regime == "storm"

        # calm: night, low wind
        fv = {"fc_icon_eu_speed": 1.0, "fc_icon_eu_cape": 0}
        res = classify_regime(datetime(2026, 7, 15, 2, 0), fv)
        assert res.regime == "calm"
