from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# df1 = df.loc[0:,['Rank', 'Name', 'City', 'State', 'category','Student_Population','Total_Annual_Cost']]
# df1["Rank"] = df1["Rank"].astype("int")

# cat_cols = df1.select_dtypes(["O"]).keys()

# for var in cat_cols:
#     df1[var].fillna(df1[var].mode()[0], inplace=True)

# df2 = df1["Name"]
# df2 = pd.DataFrame({"Name":df2})
# df3 = df1["City"]
# df3 = pd.DataFrame({"City":df3})
# df4 = df1["State"]
# df4 = pd.DataFrame({"State":df4})

# value = df1["Name"].value_counts().keys()
# value7 = df1["Name"].value_counts().keys()
# value1 = df1["City"].value_counts().keys()
# value2 = df1["State"].value_counts().keys()
# value3 = df1["category"].value_counts().keys()

# for num,var in enumerate(value):
#     num+=1
#     df1["Name"].replace(var, num, inplace=True)

# for num, var in enumerate(value1):
#     num+=1
#     df1["City"].replace(var, num, inplace=True)

# for num,var in enumerate(value2):
#     num+=1
#     df1["State"].replace(var, num, inplace=True)

# for num,var in enumerate(value3):
#     num+=1
#     df1["category"].replace(var, num, inplace=True)

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

sc=StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("heart_failure_record_prediction_model.pkl")

def heart_failure_record_prediction_model(model, age, anaemia, creatinine_phosphokinase, diabetes,
       ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    
            
    x = np.zeros(len(X.columns))
    
    x[0] = age
    x[1] = anaemia
    x[2] = creatinine_phosphokinase
    x[3] = diabetes
    x[4] = ejection_fraction
    x[5] = high_blood_pressure
    x[6] = platelets
    x[7] = serum_creatinine
    x[8] = serum_sodium
    x[9] = sex
    x[10] = smoking
    x[5] = time
    
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    # value7 = list(df2["Name"].value_counts().keys())
    # value7.sort()
    # value10 = list(df3["City"].value_counts().keys())
    # value10.sort()
    # value11 = list(df4["State"].value_counts().keys())
    # value11.sort()
    return render_template("index.html")    #,value=value7, value01=value10,value02=value11)

@app.route("/predict", methods=["POST"])
def predict():
    age = request.form["age"]
    anaemia = request.form["anaemia"]
    creatinine_phosphokinase = request.form["creatinine_phosphokinase"]
    diabetes = request.form["diabetes"]
    ejection_fraction = request.form["ejection_fraction"]
    high_blood_pressure = request.form["high_blood_pressure"]
    platelets = request.form["platelets"]
    serum_creatinine = request.form["serum_creatinine"]
    serum_sodium = request.form["serum_sodium"]
    sex = request.form["sex"]
    smoking = request.form["smoking"]
    time = request.form["time"]
    
    predicated_price = heart_failure_record_prediction_model(model, age, anaemia, creatinine_phosphokinase, diabetes,
                                                            ejection_fraction, high_blood_pressure, platelets, serum_creatinine, 
                                                            serum_sodium, sex, smoking, time)
    if predicated_price==1:
        return render_template("index.html", prediction_text="patient has heart fail")
    else:
        return render_template("index.html", prediction_text="patient will safe keep enjoy")


if __name__ == "__main__":
    app.run()    
    