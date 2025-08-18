from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# -------------------------
# Configuración de Spark
# -------------------------
try:
    spark.stop()
except:
    pass

spark = SparkSession.builder.appName("RobosPeru").getOrCreate()

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

# -------------------------
# Modalidades frecuentes
# -------------------------
modalidades = df.groupBy("UBIGEO_HECHO", "P_MODALIDADES") \
                .agg(F.sum("cantidad").alias("total")) \
                .orderBy(F.desc("total"))

# -------------------------
# Zonas más reportadas
# -------------------------
zonas = df.groupBy("UBIGEO_HECHO", "DIST_HECHO") \
          .agg(F.sum("cantidad").alias("total")) \
          .orderBy(F.desc("total"))

# -------------------------
# Predicción temporal (estacionalidad)
# -------------------------
temporal = df.groupBy("MES") \
             .agg(F.sum("cantidad").alias("total")) \
             .orderBy("MES")

# -------------------------
# Historial de riesgo
# -------------------------
historial = df.groupBy("ANIO", "MES", "UBIGEO_HECHO") \
              .agg(F.sum("cantidad").alias("total")) \
              .orderBy("ANIO", "MES")

# -------------------------
# Guardar resultados en PostgreSQL
# -------------------------
jdbc_url = "jdbc:postgresql://postgres:5432/robosdb"
db_properties = {
    "user": "admin",
    "password": "admin123",
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
