FROM python:3.8-slim

# Install dependencies for FSL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gzip \
    tar \   
    && rm -rf /var/lib/apt/lists/*

# Download and install FSL
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    python3 fslinstaller.py --dest=/opt/fsl -V 6.0.7 && \
    rm fslinstaller.py

# Set environment variables for FSL
ENV FSLDIR=/opt/fsl
RUN . /opt/fsl/etc/fslconf/fsl.sh
ENV PATH=/opt/fsl/share/fsl/bin:$PATH
RUN echo $FSLDIR
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Verify FSL installation
RUN echo $FSLDIR
RUN fsl &

# Set the working directory in the container
WORKDIR /app
COPY requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install --root-user-action=ignore --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Copy the current directory contents into the container at /app
COPY . /app
RUN rm -rf inputs/fsl/

# RUN fslswapdim
# RUN fslreorient2std
# RUN fslorient

# Command to run your application
# CMD uvicorn app:app --host 127.0.0.1 --port 1239
CMD ["python3", "seg_fcd_test.py"]