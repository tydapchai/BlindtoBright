from flask import Flask, jsonify, request, render_template_string
from config import state
import time
import logging

logger = logging.getLogger("web-app")
web = Flask(__name__)

# Giữ nguyên chuỗi HTML của bạn ở đây
HTML = """
<!doctype html>
<html lang="vi">
<!-- (Copy nguyên cục HTML cũ của bạn vào đây) -->
<h1>Blind to Bright - Web Dashboard</h1>
<div id="history"></div>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/history');
    const data=await r.json();
    document.getElementById('history').innerHTML=data.slice().reverse().map(x=>{
      return `<div><b>${x.source}</b>: ${x.text} <small>${x.time}</small></div>`;
    }).join('');
  }catch(e){}
}
refresh(); setInterval(refresh,2000);
</script>
</html>
"""

@web.get("/")
def home():
    return render_template_string(HTML)

@web.get("/api/history")
def history():
    return jsonify(state.get())

def run_web(host, port):
    try:
        from waitress import serve
        logger.info(f"Web server on http://{host}:{port}")
        serve(web, host=host, port=port, threads=4)
    except ImportError:
        web.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)