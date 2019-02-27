import json
from flask import Flask,jsonify,request,render_template
import Classifier as cl
import FB_Model as fb

# print(fb.test())

# print("Score : " + str(cl.get_score()))

app = Flask(__name__)

@app.route("/api/tags",methods=["GET"])
def func():
    Title=request.args.get('title')
    Body=request.args.get('body')
    t=str(Title) + str(Body)
    txt=cl.stemm_stop(cl.clean(t))
    x=cl.TFIDF.transform([txt]).toarray()
    res=cl.lr.predict(x)[0]
    Tags=list(fb.get_ferq_with_txt(txt,[res]))
    data={'title':Title,'body':Body,'tags':Tags}
    return jsonify(data)



#################################### For solving cross ##########################
@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

###################################  Runnting the server #################################################
if __name__ == '__main__':
    app.run(host="127.0.0.1",port=9090)
