FROM python:3.6-slim-stretch

WORKDIR /app

RUN pip3 install matplotlib numpy scipy patsy pandas statsmodels
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

CMD ["python", "-u", "main.py"]
