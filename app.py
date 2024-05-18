from flask import Flask,request,render_template,jsonify
from prediction_service import prediction

application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():    
    if request.method == "POST":
        try:
            if request.form:
                print("request is:",request)
                response = prediction.form_response(request)
                return render_template('results.html',final_result=response)
            elif request.json:
                response = prediction.api_response(request)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("form.html")






if __name__=="__main__":
    app.run(host='0.0.0.0', port = 8080,debug=True)