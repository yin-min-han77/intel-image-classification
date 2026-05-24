from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('intel_classifier.h5')
CLASS_NAMES = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
CLASS_ICONS = ['🏙️', '🌲', '🧊', '⛰️', '🌊', '🛣️']

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scene Classifier</title>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Nunito', sans-serif;
    background: #f0f4ff;
    display: flex;
    min-height: 100vh;
  }

  /* Sidebar */
  .sidebar {
    width: 220px;
    min-height: 100vh;
    background: #1e2a4a;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 36px 20px 24px;
    gap: 20px;
    flex-shrink: 0;
  }
  .school-logo {
    width: 82px; height: 82px;
    border-radius: 50%;
    background: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 38px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
  }
  .school-name {
    color: #a8b8e0;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: center;
    line-height: 1.6;
  }
  .divider { width: 100%; height: 1px; background: rgba(255,255,255,0.08); }
  .student-block { text-align: center; }
  .s-label { font-size: 10px; color: #4a5f8a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
  .s-name  { color: #ffffff; font-size: 15px; font-weight: 700; line-height: 1.4; }
  .s-id    { color: #6a80aa; font-size: 12px; margin-top: 4px; }
  .sidebar-footer { margin-top: auto; text-align: center; color: #2e3f6a; font-size: 11px; line-height: 1.7; }

  /* Main */
  .main {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 32px;
  }
  .card {
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 4px 24px rgba(30,42,74,0.10);
    padding: 36px;
    width: 100%;
    max-width: 500px;
  }
  .card-title { font-size: 22px; font-weight: 800; color: #1e2a4a; margin-bottom: 4px; }
  .card-sub   { font-size: 13px; color: #8898bb; margin-bottom: 28px; }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed #c5d0e8;
    border-radius: 14px;
    background: #f7f9ff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 40px 20px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    min-height: 200px;
  }
  .upload-zone:hover  { border-color: #4a6cf7; background: #eef1ff; }
  .upload-zone.filled { padding: 0; border-style: solid; border-color: #4a6cf7; }
  .upload-zone.filled .placeholder { display: none; }
  .upload-zone img { display: none; width: 100%; max-height: 240px; object-fit: contain; border-radius: 12px; }
  .upload-zone.filled img { display: block; }
  .placeholder { display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .up-icon { font-size: 40px; }
  .up-text { font-size: 14px; font-weight: 700; color: #3a4a70; }
  .up-hint { font-size: 12px; color: #8898bb; }
  #fileInput { display: none; }

  /* Buttons */
  .btn-row { display: flex; gap: 10px; margin-top: 16px; }
  .btn {
    flex: 1; padding: 12px;
    border-radius: 10px;
    font-family: 'Nunito', sans-serif;
    font-size: 14px; font-weight: 700;
    border: none; cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
  }
  .btn:hover   { opacity: 0.88; transform: translateY(-1px); }
  .btn:active  { transform: translateY(0); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-primary { background: #4a6cf7; color: #fff; }
  .btn-ghost   { background: #f0f4ff; color: #4a6cf7; border: 1.5px solid #c5d0e8; }

  /* Result box */
  .result-box {
    margin-top: 22px;
    border-radius: 14px;
    background: #f7f9ff;
    border: 1.5px solid #e0e8ff;
    padding: 20px;
    min-height: 80px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .result-placeholder { color: #b0bcd8; font-size: 13px; text-align: center; margin: auto; }

  .result-main { display: flex; align-items: center; gap: 14px; }
  .r-emoji { font-size: 42px; }
  .r-label { font-size: 11px; color: #8898bb; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
  .r-class { font-size: 22px; font-weight: 800; color: #1e2a4a; }
  .r-conf  { font-size: 13px; color: #4a6cf7; font-weight: 600; margin-top: 2px; }

  .bars { display: flex; flex-direction: column; gap: 7px; }
  .bar-row { display: grid; grid-template-columns: 90px 1fr 42px; align-items: center; gap: 8px; }
  .bar-name  { font-size: 12px; font-weight: 600; color: #3a4a70; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar-track { height: 7px; background: #e8eeff; border-radius: 99px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 99px; background: #c5d0e8; transition: width 0.5s ease; }
  .bar-fill.top { background: #4a6cf7; }
  .bar-pct   { font-size: 11px; color: #8898bb; text-align: right; }

  .spinner {
    display: none; width: 24px; height: 24px;
    border: 3px solid #c5d0e8; border-top-color: #4a6cf7;
    border-radius: 50%; animation: spin 0.7s linear infinite; margin: auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<aside class="sidebar">
 <div class="school-logo">
  <img src="photo_2026-05-23_11-14-49.jpg" style="width:100%;height:100%;border-radius:50%;object-fit:cover;">
</div>
  <div class="school-name">Parami University</div>
  <div class="divider"></div>
  <div class="student-block">
    <div class="s-label">Student</div>
    <div class="s-name">Yin Min Han</div>
    <div class="s-id">ID: PIUS20230069</div>
  </div>
  <div class="divider"></div>
  <div class="student-block">
    <div class="s-label">Subject</div>
    <div class="s-name" style="font-size:13px">Machine Learning<br>Final Project</div>
  </div>
  <div class="sidebar-footer">
    Intel Image Classification<br>CNN · 2025
  </div>
</aside>

<main class="main">
  <div class="card">
    <div class="card-title">Scene Classifier 🔍</div>
    <div class="card-sub">Upload a photo to identify the scene type</div>

    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
      <div class="placeholder">
        <div class="up-icon">🖼️</div>
        <div class="up-text">Click to upload an image</div>
        <div class="up-hint">JPG or PNG supported</div>
      </div>
      <img id="preview" src="" alt="preview">
      <input type="file" id="fileInput" accept="image/*">
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost"   onclick="clearAll()">Clear</button>
      <button class="btn btn-primary" id="classifyBtn" onclick="classify()" disabled>Classify</button>
    </div>

    <div class="result-box" id="resultBox">
      <div class="result-placeholder" id="placeholder">Result will appear here after classification</div>
      <div class="spinner" id="spinner"></div>
      <div id="resultContent" style="display:none">
        <div class="result-main">
          <div class="r-emoji" id="rEmoji"></div>
          <div>
            <div class="r-label">Predicted Scene</div>
            <div class="r-class" id="rClass"></div>
            <div class="r-conf"  id="rConf"></div>
          </div>
        </div>
        <div class="bars" id="rBars"></div>
      </div>
    </div>
  </div>
</main>

<script>
  let file = null;

  document.getElementById('fileInput').addEventListener('change', function(e) {
    const f = e.target.files[0];
    if (!f) return;
    file = f;
    const reader = new FileReader();
    reader.onload = ev => {
      document.getElementById('preview').src = ev.target.result;
      document.getElementById('uploadZone').classList.add('filled');
      document.getElementById('classifyBtn').disabled = false;
      reset();
    };
    reader.readAsDataURL(f);
  });

  function clearAll() {
    file = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('preview').src = '';
    document.getElementById('uploadZone').classList.remove('filled');
    document.getElementById('classifyBtn').disabled = true;
    reset();
  }

  function reset() {
    document.getElementById('placeholder').style.display = 'block';
    document.getElementById('resultContent').style.display = 'none';
    document.getElementById('spinner').style.display = 'none';
  }

  async function classify() {
    if (!file) return;
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('resultContent').style.display = 'none';
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('classifyBtn').disabled = true;

    const form = new FormData();
    form.append('image', file);

    try {
      const res  = await fetch('/predict', { method: 'POST', body: form });
      const data = await res.json();
      if (data.error) { alert(data.error); reset(); return; }

      document.getElementById('rEmoji').textContent = data.icon;
      document.getElementById('rClass').textContent = data.predicted_class;
      document.getElementById('rConf').textContent  = 'Confidence: ' + data.confidence + '%';

      document.getElementById('rBars').innerHTML = data.scores.map((s, i) => `
        <div class="bar-row">
          <span class="bar-name">${s.icon} ${s.name}</span>
          <div class="bar-track"><div class="bar-fill ${i===0?'top':''}" style="width:${s.score}%"></div></div>
          <span class="bar-pct">${s.score}%</span>
        </div>`).join('');

      document.getElementById('resultContent').style.display = 'block';
    } catch(e) {
      alert('Something went wrong. Please try again.');
      reset();
    } finally {
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('classifyBtn').disabled = false;
    }
  }
</script>
</body>
</html>
"""


@app.route('/')
def home():
    return HTML


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((150, 150))
    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)

    preds = model.predict(img_array, verbose=0)[0].tolist()
    top_idx = int(np.argmax(preds))

    scores = sorted([
        {'name': CLASS_NAMES[i], 'icon': CLASS_ICONS[i], 'score': round(preds[i] * 100, 1)}
        for i in range(len(CLASS_NAMES))
    ], key=lambda x: x['score'], reverse=True)

    return jsonify({
        'predicted_class': CLASS_NAMES[top_idx],
        'icon':            CLASS_ICONS[top_idx],
        'confidence':      round(preds[top_idx] * 100, 1),
        'scores':          scores
    })


if __name__ == '__main__':
    app.run(debug=True)
