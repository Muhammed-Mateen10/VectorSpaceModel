from flask import Flask, render_template, request
from VectorSpaceModel import VSM
app = Flask(__name__)

VSM = VSM()

@app.route("/")
def hello():
  return render_template('home.html')

# @app.route("/search")
# def 

@app.route("/help")
def help():
  return render_template("help.html")

@app.route("/search" , methods= ["GET" , "POST"])
def gfg():
  if request.method == 'POST':
    input = request.form.get("Query")
    matchedDocs = VSM.runQuery(input)
    return render_template("search.html" , input = input , uresult = matchedDocs[0] , rresult = matchedDocs[1])




if __name__ == "__main__":
  app.run()