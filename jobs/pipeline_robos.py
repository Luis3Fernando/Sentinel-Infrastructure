from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id

# -------------------------
# Configuración de Spark
# -------------------------
try:
    spark.stop()
except:
    pass

spark = SparkSession.builder \
    .appName("Pipeline Robos") \
    .config("spark.jars", "/libs/postgresql-42.6.0.jar") \
    .getOrCreate()

# -------------------------
# Lectura de datos
# -------------------------
df = spark.read.csv("/data/robos.csv", header=True, inferSchema=True)

# -------------------------
# Consulta de riesgo por distrito
# -------------------------
riesgo = df.groupBy("UBIGEO_HECHO", "DIST_HECHO") \
           .agg(F.sum("cantidad").alias("total_robos"))

max_total = riesgo.agg(F.max("total_robos")).collect()[0][0]

riesgo = riesgo.withColumn(
    "nivel_riesgo",
    F.when(F.col("total_robos") < 0.33 * max_total, "Bajo")
     .when(F.col("total_robos") < 0.66 * max_total, "Medio")
     .otherwise("Alto")
)

# Agregar id
riesgo = riesgo.withColumn("id", monotonically_increasing_id())

# -------------------------
# Modalidades frecuentes
# -------------------------
modalidades = df.groupBy("UBIGEO_HECHO", "P_MODALIDADES") \
                .agg(F.sum("cantidad").alias("total")) \
                .orderBy(F.desc("total"))

modalidades = modalidades.withColumn("id", monotonically_increasing_id())

# -------------------------
# Zonas más reportadas
# -------------------------
zonas = df.groupBy("UBIGEO_HECHO", "DIST_HECHO") \
          .agg(F.sum("cantidad").alias("total")) \
          .orderBy(F.desc("total"))

zonas = zonas.withColumn("id", monotonically_increasing_id())

# -------------------------
# Predicción temporal (estacionalidad)
# -------------------------
temporal = df.groupBy("MES") \
             .agg(F.sum("cantidad").alias("total")) \
             .orderBy("MES")

temporal = temporal.withColumn("id", monotonically_increasing_id())

# -------------------------
# Historial de riesgo
# -------------------------
historial = df.groupBy("ANIO", "MES", "UBIGEO_HECHO") \
              .agg(F.sum("cantidad").alias("total")) \
              .orderBy("ANIO", "MES")

historial = historial.withColumn("id", monotonically_increasing_id())

# -------------------------
# Guardar resultados en PostgreSQL
# -------------------------
jdbc_url = "jdbc:postgresql://postgres:5432/robos_db"  # 👀 usa 5432 interno, no el 15432 del host
db_properties = {
    "user": "robos_user",
    "password": "robos_pass",
    "driver": "org.postgresql.Driver"
}

riesgo.write.jdbc(url=jdbc_url, table="riesgo", mode="overwrite", properties=db_properties)
modalidades.write.jdbc(url=jdbc_url, table="modalidades", mode="overwrite", properties=db_properties)
zonas.write.jdbc(url=jdbc_url, table="zonas", mode="overwrite", properties=db_properties)
temporal.write.jdbc(url=jdbc_url, table="temporal", mode="overwrite", properties=db_properties)
historial.write.jdbc(url=jdbc_url, table="historial", mode="overwrite", properties=db_properties)

# -------------------------
# Finalizar Spark
# -------------------------
spark.stop()
