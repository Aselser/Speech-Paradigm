from flask import Flask, render_template, request
import csv
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Cambia al directorio padre (paradigma)
        os.chdir('..')

        # Collect form data
        form_data = request.form

        # Construye la ruta al archivo CSV en data
        csv_file_path = os.path.join(os.getcwd(),'data', 'data.csv')

        # Write form data to CSV file
        write_to_csv(csv_file_path, form_data)

        return render_template('success.html')

    finally:
        os.chdir('formulario')


def write_to_csv(file_path, data):
    # Check if the CSV file exists, create it if not
    is_file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='',encoding='utf-8') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers only if the file is created now
        if not is_file_exists:
            writer.writeheader()

        writer.writerow(data)

if __name__ == '__main__':
    app.run(debug=True)
