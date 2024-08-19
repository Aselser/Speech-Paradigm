import csv
import os
import webbrowser
from flask import Flask, render_template, request
from threading import Timer

class Formulario:
    def __init__(self, template_folder='formulario/templates', static_folder='formulario/static'):
        self.app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
        self._add_routes()

    def _add_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/submit', methods=['POST'])
        def submit():
            try:
                # Change to the parent directory (paradigma)
                os.chdir('..')

                # Collect form data
                form_data = request.form

                # Construct the path to the CSV file in data
                csv_file_path = os.path.join(os.getcwd(), 'data', 'data.csv')

                # Write form data to CSV file
                self.write_to_csv(csv_file_path, form_data)

                return render_template('success.html')

            finally:
                os.chdir('formulario')

    def write_to_csv(self, file_path, data):
        # Check if the CSV file exists, create it if not
        is_file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = data.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write headers only if the file is created now
            if not is_file_exists:
                writer.writeheader()

            writer.writerow(data)

    def open_browser(self):
        webbrowser.open_new("http://127.0.0.1:5000/")

    def run(self, debug=True):
        # Start the Flask app in a new thread
        Timer(1, self.open_browser).start()
        self.app.run(debug=debug)

if __name__ == "__main__":
    my_app = Formulario()
    my_app.run()