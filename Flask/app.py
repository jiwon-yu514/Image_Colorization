from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
import os
from werkzeug.utils import secure_filename
from colorize_model import colorize_image

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == 'konyang001' and password == 'konyang001!':
        session['username'] = username
        return redirect(url_for('dashboard'))
    else:
        flash("아이디 또는 비밀번호가 틀렸습니다.")
        return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        result_files = colorize_image(input_path, OUTPUT_FOLDER, filename)

        # 세션에 업로드 기록 저장
        session.setdefault('history', [])
        session['history'].append({
            'input': filename,
            'results': result_files
        })
        session.modified = True

        return render_template('result.html', result_files=result_files, filename=filename)
    return redirect(url_for('dashboard'))


@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/uploads/<filename>')
def send_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/final_select', methods=['POST'])
def final_select():
    selected_image = request.form['selected_image']
    flash(f"최종 선택한 결과: {selected_image}")
    return render_template('final_result.html', selected_image=selected_image)


@app.route('/download/<filename>')
def download_result(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route('/share/<filename>')
def share_result(filename):
    return render_template('share.html', filename=filename)


@app.route('/logout')
def logout():
    session.clear()
    flash("로그아웃되었습니다.")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
