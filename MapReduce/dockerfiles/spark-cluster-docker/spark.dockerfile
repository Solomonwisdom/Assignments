FROM solomonfield/hadoop:2.7.7
LABEL Solomonfield <whg19961229@gmail.com>

ENV SCALA_VERSION=2.11.8
ENV SPARK_VERSION=2.3.3
# Download first
RUN wget https://downloads.lightbend.com/scala/${SCALA_VERSION}/scala-${SCALA_VERSION}.rpm && \
    wget https://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
	rpm -i scala-${SCALA_VERSION}.rpm && \
    rm scala-${SCALA_VERSION}.rpm && \
	tar xvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
	mv spark-${SPARK_VERSION}-bin-hadoop2.7 /usr/local/spark && \
	rm -rf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz
	
COPY config/* /tmp/

RUN cp $HADOOP_HOME/etc/hadoop/slaves /usr/local/spark/conf/slaves && \
	echo "export SCALA_HOME=/usr/share/scala" >> /usr/local/spark/conf/spark-env.sh && \
	echo "export JAVA_HOME=$JAVA_HOME" >> /usr/local/spark/conf/spark-env.sh && \
	echo "export HADOOP_HOME=/usr/local/hadoop" >> /usr/local/spark/conf/spark-env.sh && \
	echo "export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop" >> /usr/local/spark/conf/spark-env.sh && \
	echo "SPARK_MASTER_IP=master" >> /usr/local/spark/conf/spark-env.sh && \
	echo "SPARK_LOCAL_DIRS=/usr/local/spark" >> /usr/local/spark/conf/spark-env.sh && \
	echo "SPARK_DRIVER_MEMORY=4G" >> /usr/local/spark/conf/spark-env.sh && \
	echo "SPARK_HOME=/usr/local/spark" >> /usr/local/spark/conf/spark-env.sh && \
	echo 'export SPARK_HISTORY_OPTS="-Dspark.history.ui.port=18080 -Dspark.history.retainedApplications=50 -Dspark.history.fs.logDirectory=hdfs://master:9000/spark/history"' >> /usr/local/spark/conf/spark-env.sh && \
    echo "spark.eventLog.enabled    true" >> /usr/local/spark/conf/spark-default.conf && \
	echo "spark.eventLog.dir        hdfs://master:9000/spark/history" >> /usr/local/spark/conf/spark-default.conf && \
	echo "spark.eventLog.compress   true" >> /usr/local/spark/conf/spark-default.conf && \
	echo "spark.driver.memory       4g" >> /usr/local/spark/conf/spark-default.conf && \
	mv /tmp/spark-entrypoint.sh /root/spark-entrypoint.sh

ENV SPARK_HOME /usr/local/spark
ENV PATH $PATH:$SPARK_HOME/bin
ENV HIVE_SKIP_SPARK_ASSEMBLY=true
EXPOSE 8080 4042 4040 4041
ENTRYPOINT [ "sh", "-c", "/root/spark-entrypoint.sh; bash"]
