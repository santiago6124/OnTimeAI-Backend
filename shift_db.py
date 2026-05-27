import sqlite3
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from ontimeai.live import open_db

def run():
    # Trigger database migrations first
    con = open_db(Path("live_data.db"))
    cursor = con.cursor()

    row = cursor.execute("SELECT MAX(fl_date) FROM flights").fetchone()
    if not row or not row[0]:
        print("No flights found in database")
        con.close()
        return
        
    max_date_str = row[0]
    max_date = datetime.strptime(max_date_str, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    delta_days = (today - max_date).days

    if delta_days == 0:
        print("Database is already up to date (today's date).")
        con.close()
        return

    print(f"Shifting database from {max_date_str} to {today} (delta: {delta_days} days)...")

    def shift_iso_time(val):
        if not val:
            return val
        try:
            # handle Z suffix
            original_z = val.endswith("Z")
            if original_z:
                val = val[:-1] + "+00:00"
            dt = datetime.fromisoformat(val)
            dt_shifted = dt + timedelta(days=delta_days)
            res = dt_shifted.isoformat()
            if original_z and "+00:00" in res:
                res = res.replace("+00:00", "Z")
            return res
        except Exception as e:
            return val

    con.create_function("shift_time", 1, shift_iso_time)

    # 1. Update flights
    cursor.execute("UPDATE flights SET fl_date = date(fl_date, '+' || ? || ' days')", (delta_days,))
    cursor.execute("UPDATE flights SET scheduled_out_utc = shift_time(scheduled_out_utc)")
    cursor.execute("UPDATE flights SET scheduled_off_utc = shift_time(scheduled_off_utc)")
    cursor.execute("UPDATE flights SET scheduled_on_utc = shift_time(scheduled_on_utc)")
    cursor.execute("UPDATE flights SET scheduled_in_utc = shift_time(scheduled_in_utc)")
    cursor.execute("UPDATE flights SET estimated_out_utc = shift_time(estimated_out_utc)")
    cursor.execute("UPDATE flights SET estimated_in_utc = shift_time(estimated_in_utc)")
    cursor.execute("UPDATE flights SET first_seen_utc = shift_time(first_seen_utc)")
    cursor.execute("UPDATE flights SET last_updated_utc = shift_time(last_updated_utc)")

    # 2. Update predictions (use temp table to avoid UNIQUE constraint violations)
    cursor.execute("""
        CREATE TEMP TABLE temp_predictions AS 
        SELECT fa_flight_id, stable_id, shift_time(predicted_at_utc) AS predicted_at_utc, 
               proba_delay, predicted_delay, threshold_used, threshold_strategy
        FROM predictions
    """)
    cursor.execute("DELETE FROM predictions")
    cursor.execute("""
        INSERT OR REPLACE INTO predictions 
        (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay, threshold_used, threshold_strategy)
        SELECT * FROM temp_predictions
    """)
    cursor.execute("DROP TABLE temp_predictions")

    # 3. Update actuals
    cursor.execute("UPDATE actuals SET actual_out_utc = shift_time(actual_out_utc)")
    cursor.execute("UPDATE actuals SET actual_off_utc = shift_time(actual_off_utc)")
    cursor.execute("UPDATE actuals SET actual_on_utc = shift_time(actual_on_utc)")
    cursor.execute("UPDATE actuals SET actual_in_utc = shift_time(actual_in_utc)")
    cursor.execute("UPDATE actuals SET settled_at_utc = shift_time(settled_at_utc)")

    # 4. Update weather_obs (use temp table to avoid UNIQUE constraint violations)
    cursor.execute("""
        CREATE TEMP TABLE temp_weather AS 
        SELECT station, shift_time(valid_utc) AS valid_utc, tmpc, dwpc, relh, drct, sknt, alti,
               p01m, vsby, gust, wxcodes, wx_precip_flag, wx_low_vis_flag, wx_strong_wind_flag
        FROM weather_obs
    """)
    cursor.execute("DELETE FROM weather_obs")
    cursor.execute("""
        INSERT OR REPLACE INTO weather_obs
        (station, valid_utc, tmpc, dwpc, relh, drct, sknt, alti, p01m, vsby, gust, wxcodes,
         wx_precip_flag, wx_low_vis_flag, wx_strong_wind_flag)
        SELECT * FROM temp_weather
    """)
    cursor.execute("DROP TABLE temp_weather")

    con.commit()
    con.close()
    print("✓ Database shifted successfully!")

if __name__ == "__main__":
    run()
