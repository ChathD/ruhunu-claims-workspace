"""Check historical lookup boundaries using the provider's public chart format."""
import json
import unittest
from unittest.mock import patch

import httpx

from backend.evidence import vehicle_market_value as market


def chart(rows, make="toyota", model="aqua"):
    data = json.dumps({"initialMake": make, "initialModel": model, "initialTrend": rows})
    html = f"<script>self.__next_f.push({json.dumps([1, data])})</script>"
    return httpx.Response(200, text=html, request=httpx.Request("GET", market.PRICE_TREND_URL))


class VehicleMarketValueTests(unittest.TestCase):
    def test_repair_cap_assumption_needs_a_recent_benchmark_with_room_for_variation(self):
        claim = {"gross_amount": "187500", "loss_date": "2026-11-25"}
        benchmark = {"status": "found", "currency": "LKR", "asking_price_lkr": "8495000", "observation_date": "2026-11-21",
                     "source": "PriceMart.lk", "make": "Toyota", "model": "Vitz", "year": 2019}
        def assess(**changes):
            return market.assess_repair_market_cap(claim, {"result": {**benchmark, **changes}, "evidence_ref": "tool:market"})
        self.assertEqual(assess()["status"], "unlikely_to_limit")
        for changes in ({"asking_price_lkr": "190000"}, {"observation_date": "2025-11-21"}, {"status": "unavailable"}):
            with self.subTest(changes=changes):
                self.assertEqual(assess(**changes)["status"], "unconfirmed")

    def test_repair_market_cap_excludes_proportional_reduction_and_required_valuations(self):
        claim = {"loss_type": "repair", "currency": "LKR", "loss_date": "2026-11-25", "registration": "TEST-1234"}
        benefit = {"loss_type": "repair", "limit_basis": "market_value"}
        schedule = {"starts_on": "2026-10-20", "ends_on": "2027-10-19", "vehicle": {"registration": "TEST-1234"}, "cover": [benefit]}
        self.assertTrue(market.repair_market_cap_applies(claim, schedule))
        for term in ("A condition of average applies.", "The claimant must provide an independent valuation."):
            with self.subTest(term=term):
                benefit["conditions"] = term
                self.assertFalse(market.repair_market_cap_applies(claim, schedule))

    def test_exact_year_and_latest_observation_at_or_before_requested_date(self):
        rows = [{"date": "2026-11-13", "2019": 13000000},
                {"date": "2026-08-29", "2019": 12900000},
                {"date": "2026-11-12", "2018": 9700000}]
        query = market.VehicleMarketQuery(make="Toyota", model="Aqua", year=2019, as_of="12/11/2026")
        with patch.object(market.httpx, "get", return_value=chart(rows)) as fetch:
            result = market.lookup_vehicle_market_value(query)
            exact = market.lookup_vehicle_market_value(query.model_copy(update={"year": 2018}))
        fetch.assert_called_with("https://pricemart.lk/dashboard/trend/toyota/aqua", timeout=8)
        self.assertEqual((result["status"], result["asking_price_lkr"], result["observation_date"], result["days_before_requested"]),
                         ("found", "12900000", "2026-08-29", 75))
        self.assertEqual((exact["asking_price_lkr"], exact["days_before_requested"]), ("9700000", 0))

    def test_missing_observation_and_mismatched_provider_page_are_distinct(self):
        rows = [{"date": "2026-11-13", "2019": 13000000}]
        query = market.VehicleMarketQuery(make="Toyota", model="Aqua", year=2019, as_of="2026-11-12")
        with patch.object(market.httpx, "get", side_effect=[chart(rows), chart(rows, model="vitz")]):
            self.assertEqual(market.lookup_vehicle_market_value(query)["status"], "no_observation")
            self.assertEqual(market.lookup_vehicle_market_value(query)["status"], "unavailable")
