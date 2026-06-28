# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /code

# Copy and install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user for security (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set up application folder
WORKDIR $HOME/app

# Copy application files and set ownership
COPY --chown=user . $HOME/app

# HuggingFace Spaces expects containers to listen on port 7860
EXPOSE 7860

# Run Streamlit on port 7860
ENTRYPOINT ["streamlit", "run", "sandbox/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
