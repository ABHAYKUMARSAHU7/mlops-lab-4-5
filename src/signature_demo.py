import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:///mlflow.db")
m = mlflow.pyfunc.load_model("models:/HousePriceModel@production")
good = {"area_sqft":[1500.0],"bedrooms":[3.0],"age_years":[10.0],"distance_to_city_km":[5.0]}
tests = {
 "valid request": good,
 "3 of 4 columns": {k:v for k,v in good.items() if k!="distance_to_city_km"},
 "string value": {**good,"area_sqft":["big"]},
 "ints instead of doubles": {**good,"area_sqft":[1500],"bedrooms":[3]},
 "extra column": {**good,"zipcode":[380001]},
 "nonsense values": {**good,"area_sqft":[-5000.0],"age_years":[9999.0]},
}
for n,d in tests.items():
    try: print(f"[OK ] {n}: {m.predict(pd.DataFrame(d))}")
    except Exception as e:
        msg = " ".join(str(e).split())
        reason = msg.split("Error:", 1)[-1].strip()
        print(f"[ERR] {n}: {reason[:230]}")
