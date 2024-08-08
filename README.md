# README 

## Initial Setup

### On macOS/Linux

1. Open the repository directory and create a virtual environment for Python:
    ```bash
    cd laser_ablation_scripts
    python3 -m venv venv
    ```

2. Activate the virtual environment (this must be done before running the code or using `pip`):
    ```bash
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Optionally, install additional packages for enhanced functionality:
    ```bash
    pip install pyqtgraph pyside6
    ```

### On Windows

1. Open the repository directory and create a virtual environment for Python:
    ```cmd
    cd laser_ablation_scripts
    python -m venv venv
    ```

2. Activate the virtual environment (this must be done before running the code or using `pip`):
    ```cmd
    venv\Scripts\activate
    ```

3. Install the required packages:
    ```cmd
    pip install -r requirements.txt
    ```

4. Optionally, install additional packages for enhanced functionality:
    ```cmd
    pip install pyqtgraph pyside6
    ```
## License

This project is licensed under a dual-license model:

- **GNU General Public License v2.0 (GPLv2):** You may redistribute and/or modify this software under the terms of the GPLv2 as published by the Free Software Foundation. See the `LICENSE` file for details.

- **Commercial License:** For those who prefer to use this software under a closed-source or commercial license, a separate license is available. Please contact [Serva Energy](mailto:info@servaenergy.com) for more details.

By using this software, you agree to the terms and conditions of either the GPLv2 or the commercial license.
