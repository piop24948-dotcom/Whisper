import pandas as pd # type: ignore
import re
import os
import glob
from datetime import datetime
import numpy as np

class DecodificadorLlamadas:
    def __init__(self):
        self.patrones = {
            'exten': r'exten-(\d+)-(\d+)-(\d{8})-(\d{6})-(\d+\.\d+)',
            'rg_interna': r'rg-(\d+)-(\d+)-(\d{8})-(\d{6})-(\d+\.\d+)',
            'rg_externa': r'rg-(\d+)-(\d+)-(\d{8})-(\d{6})-(\d+\.\d+)',
            'rg_anonymous': r'rg-(\d+)-anonymous-(\d{8})-(\d{6})-(\d+\.\d+)',
            'rg_alpha': r'rg-(\d+)-([a-zA-Z]+)-(\d{8})-(\d{6})-(\d+\.\d+)',
            'out': r'out-(\d+)-(\d+)-(\d{8})-(\d{6})-(\d+\.\d+)'
        }
        
        # Base de datos de extensiones y grupos
        self.extensiones = {
            # Habitacion (01-42)
            '01': 'Habitacion 1', '02': 'Habitacion 2', '03': 'Habitacion 3',
            '04': 'Habitacion 4', '05': 'Habitacion 5', '06': 'Habitacion 6',
            '07': 'Habitacion 7', '08': 'Habitacion 8', '09': 'Habitacion 9',
            '10': 'Habitacion 10', '11': 'Habitacion 11', '12': 'Habitacion 12',
            '13': 'Habitacion 13', '14': 'Habitacion 14', '15': 'Habitacion 15',
            '16': 'Habitacion 16', '17': 'Habitacion 17', '18': 'Habitacion 18',
            '19': 'Habitacion 19', '20': 'Habitacion 20', '21': 'Habitacion 21',
            '22': 'Habitacion 22', '23': 'Habitacion 23', '24': 'Habitacion 24',
            '25': 'Habitacion 25', '26': 'Habitacion 26', '27': 'Habitacion 27',
            '28': 'Habitacion 28', '29': 'Habitacion 29', '30': 'Habitacion 30',
            '31': 'Habitacion 31', '32': 'Habitacion 32', '33': 'Habitacion 33',
            '34': 'Habitacion 34', '35': 'Habitacion 35', '36': 'Habitacion 36',
            '37': 'Habitacion 37', '38': 'Habitacion 38', '39': 'Habitacion 39',
            '40': 'Habitacion 40', '41': 'Habitacion 41', '42': 'Habitacion 42',
            
            # Offchase (200s)
            '201': 'Oficina', '202': 'Cocina', '203': 'Lavanderia', 
            '200': 'recepcion', '7015': 'CDP Analog',
            
            # Moviles (7000s)
            '7001': 'Movil 1', '7002': 'Movil 2', '7003': 'Movil 3',
            '7004': 'Movil 4', '7005': 'Movil 5', '7006': 'Movil 6',
            '7007': 'Movil 7', '7008': 'Movil 8', '7009': 'Movil 9',
            '7010': 'Movil 10', '7011': 'Movil 11', '7012': 'Movil 12',
            '7013': 'Movil 13', '7014': 'Wifi 7014',
            
            # Interfones (350s)
            '354': 'hall1', '352': 'hall2',
            '255': 'Salida',
            
            # Ring Groups comunes
            '100': 'Extensión 100', '600': 'Ring Group 600',
            '253': 'Entrada',
        }
        
        self.grupos = {
            '600': 'Ring Group Oficinas',
            '601': 'Ring Group 601',  # Nuevo grupo detectado
            '700': 'Grupo Moviles',
            '351': 'Interfones'
        }
    
    def detectar_columna_nombres(self, df):
        """Detecta automáticamente qué columna contiene los nombres de archivo"""
        # Primero buscar columnas con nombres comunes
        nombres_comunes = ['archivo', 'filename', 'file_name', 'nombre_archivo', 'call_file', 'recording', 'file']
        
        for nombre in nombres_comunes:
            if nombre in df.columns:
                # Verificar que la columna contiene datos de audio
                muestra = df[nombre].dropna().head(5)
                if len(muestra) > 0 and any('.wav' in str(x) or '.mp3' in str(x) or any(patron in str(x) for patron in ['exten-', 'rg-', 'out-']) for x in muestra):
                    return nombre
        
        # Si no se encuentra, buscar por patrones en todas las columnas
        patrones_llamada = [r'exten-\d+-\d+-\d{8}-\d{6}', 
                           r'rg-\d+-\d+-\d{8}-\d{6}', 
                           r'out-\d+-\d+-\d{8}-\d{6}']
        
        for columna in df.columns:
            if df[columna].dtype == 'object':
                muestra = df[columna].dropna().head(10)
                if len(muestra) > 0:
                    coincidencias = 0
                    for valor in muestra:
                        valor_str = str(valor)
                        if any(re.search(patron, valor_str) for patron in patrones_llamada):
                            coincidencias += 1
                    
                    if coincidencias >= 3:  # Al menos 3 coincidencias
                        return columna
        
        return None
    
    def limpiar_nombre_archivo(self, nombre_archivo):
        """Limpia el nombre de archivo quitando la extensión .wav"""
        if isinstance(nombre_archivo, str):
            # Quitar extensión .wav y cualquier ruta
            nombre_limpio = os.path.basename(nombre_archivo)  # Quitar ruta
            nombre_limpio = re.sub(r'\.wav$', '', nombre_limpio)  # Quitar .wav
            nombre_limpio = re.sub(r'\.mp3$', '', nombre_limpio)  # Quitar .mp3
            return nombre_limpio
        return nombre_archivo
    
    def obtener_nombre_extension(self, numero):
        """Obtiene el nombre de la extensión basado en el número"""
        return self.extensiones.get(numero, f'Extensión {numero}')
    
    def obtener_nombre_grupo(self, numero_grupo):
        """Obtiene el nombre del grupo basado en el número"""
        return self.grupos.get(numero_grupo, f'Grupo {numero_grupo}')
    
    def determinar_tipo_numero(self, numero):
        """Determina si un número es extensión, grupo o externo"""
        if numero in self.extensiones:
            return 'Extensión'
        elif numero in self.grupos:
            return 'Grupo'
        elif len(numero) <= 3:  # Números cortos probablemente sean extensiones
            return 'Extensión'
        else:
            return 'Externo'
    
    def decodificar_nombre_archivo(self, nombre_archivo):
        """Decodifica el nombre del archivo y devuelve un diccionario con la información"""
        
        # Limpiar el nombre de archivo primero
        nombre_limpio = self.limpiar_nombre_archivo(nombre_archivo)
        
        if pd.isna(nombre_limpio) or not isinstance(nombre_limpio, str):
            return self._crear_resultado_vacio()
        
        resultado = {
            'tipo_llamada': None,
            'origen_numero': None,
            'origen_nombre': None,
            'destino_numero': None,
            'destino_nombre': None,
            'fecha': None,
            'hora': None,
            'identificador': None,
            'tipo_origen': None,
            'tipo_destino': None,
            'grupo': None,
            'grupo_nombre': None,
            'nombre_archivo_limpio': nombre_limpio
        }
        
        # Patrón para extensión interna
        if nombre_limpio.startswith('exten-'):
            match = re.match(self.patrones['exten'], nombre_limpio)
            if match:
                origen, destino, fecha, hora, identificador = match.groups()
                
                resultado.update({
                    'tipo_llamada': 'Interna',
                    'origen_numero': origen,
                    'origen_nombre': self.obtener_nombre_extension(origen),
                    'destino_numero': destino,
                    'destino_nombre': self.obtener_nombre_extension(destino),
                    'fecha': self.formatear_fecha(fecha),
                    'hora': self.formatear_hora(hora),
                    'identificador': identificador,
                    'tipo_origen': self.determinar_tipo_numero(origen),
                    'tipo_destino': self.determinar_tipo_numero(destino)
                })
                return resultado
        
        # Patrón para Ring Group con anonymous
        elif nombre_limpio.startswith('rg-') and 'anonymous' in nombre_limpio:
            match = re.match(self.patrones['rg_anonymous'], nombre_limpio)
            if match:
                grupo, fecha, hora, identificador = match.groups()
                
                resultado.update({
                    'tipo_llamada': 'Entrante Externa',
                    'origen_numero': 'anonymous',
                    'origen_nombre': 'Llamada Anónima',
                    'destino_numero': grupo,
                    'destino_nombre': self.obtener_nombre_grupo(grupo),
                    'fecha': self.formatear_fecha(fecha),
                    'hora': self.formatear_hora(hora),
                    'identificador': identificador,
                    'tipo_origen': 'Externo',
                    'tipo_destino': 'Ring Group',
                    'grupo': grupo,
                    'grupo_nombre': self.obtener_nombre_grupo(grupo)
                })
                return resultado
        
        # Patrón para Ring Group con texto en origen
        elif nombre_limpio.startswith('rg-'):
            # Primero intentar con el patrón estándar
            match_std = re.match(self.patrones['rg_interna'], nombre_limpio)
            if match_std:
                grupo, origen, fecha, hora, identificador = match_std.groups()
                
                # Determinar si es interna o externa
                if len(origen) <= 3 or origen in self.extensiones:
                    tipo_llamada = 'Interna RG'
                    tipo_origen = self.determinar_tipo_numero(origen)
                    nombre_origen = self.obtener_nombre_extension(origen)
                else:
                    tipo_llamada = 'Entrante Externa'
                    tipo_origen = 'Externo'
                    nombre_origen = f'Externo {origen}'
                
                resultado.update({
                    'tipo_llamada': tipo_llamada,
                    'origen_numero': origen,
                    'origen_nombre': nombre_origen,
                    'destino_numero': grupo,
                    'destino_nombre': self.obtener_nombre_grupo(grupo),
                    'fecha': self.formatear_fecha(fecha),
                    'hora': self.formatear_hora(hora),
                    'identificador': identificador,
                    'tipo_origen': tipo_origen,
                    'tipo_destino': 'Ring Group',
                    'grupo': grupo,
                    'grupo_nombre': self.obtener_nombre_grupo(grupo)
                })
                return resultado
            
            # Si no coincide con el patrón estándar, intentar con patrón alpha
            match_alpha = re.match(self.patrones['rg_alpha'], nombre_limpio)
            if match_alpha:
                grupo, origen_texto, fecha, hora, identificador = match_alpha.groups()
                
                resultado.update({
                    'tipo_llamada': 'Entrante Externa',
                    'origen_numero': origen_texto,
                    'origen_nombre': f'Origen: {origen_texto}',
                    'destino_numero': grupo,
                    'destino_nombre': self.obtener_nombre_grupo(grupo),
                    'fecha': self.formatear_fecha(fecha),
                    'hora': self.formatear_hora(hora),
                    'identificador': identificador,
                    'tipo_origen': 'Externo',
                    'tipo_destino': 'Ring Group',
                    'grupo': grupo,
                    'grupo_nombre': self.obtener_nombre_grupo(grupo)
                })
                return resultado
        
        # Patrón para llamada saliente
        elif nombre_limpio.startswith('out-'):
            match = re.match(self.patrones['out'], nombre_limpio)
            if match:
                destino, origen, fecha, hora, identificador = match.groups()
                
                resultado.update({
                    'tipo_llamada': 'Saliente',
                    'origen_numero': origen,
                    'origen_nombre': self.obtener_nombre_extension(origen),
                    'destino_numero': destino,
                    'destino_nombre': f'Externo {destino}',
                    'fecha': self.formatear_fecha(fecha),
                    'hora': self.formatear_hora(hora),
                    'identificador': identificador,
                    'tipo_origen': self.determinar_tipo_numero(origen),
                    'tipo_destino': 'Externo'
                })
                return resultado
        
        # Si no coincide con ningún patrón conocido
        return self._crear_resultado_vacio()
    
    def _crear_resultado_vacio(self):
        """Crea un resultado vacío para valores nulos"""
        return {
            'tipo_llamada': 'No decodificada',
            'origen_numero': None,
            'origen_nombre': None,
            'destino_numero': None,
            'destino_nombre': None,
            'fecha': None,
            'hora': None,
            'identificador': None,
            'tipo_origen': None,
            'tipo_destino': None,
            'grupo': None,
            'grupo_nombre': None,
            'nombre_archivo_limpio': None
        }
    
    def formatear_fecha(self, fecha_str):
        """Convierte fecha YYYYMMDD a formato legible"""
        try:
            return datetime.strptime(fecha_str, '%Y%m%d').strftime('%Y-%m-%d')
        except:
            return fecha_str
    
    def formatear_hora(self, hora_str):
        """Convierte hora HHMMSS a formato legible"""
        try:
            return f"{hora_str[:2]}:{hora_str[2:4]}:{hora_str[4:6]}"
        except:
            return hora_str
    
    def procesar_archivo_csv(self, archivo_csv):
        """Procesa un único archivo CSV"""
        try:
            df = pd.read_csv(archivo_csv)
            print(f"✓ Leyendo: {archivo_csv} ({len(df)} registros)")
            
        except Exception as e:
            print(f"✗ Error leyendo {archivo_csv}: {e}")
            return None
        
        # Detectar automáticamente la columna con nombres de archivo
        columna_nombres = self.detectar_columna_nombres(df)
        
        if columna_nombres is None:
            print(f"✗ No se pudo detectar la columna con nombres de archivo en {archivo_csv}")
            return None
        
        print(f"✓ Columna detectada: '{columna_nombres}'")
        
        datos_decodificados = []
        for nombre in df[columna_nombres]:
            datos = self.decodificar_nombre_archivo(nombre)
            datos['nombre_archivo_original'] = nombre
            datos['archivo_origen'] = os.path.basename(archivo_csv)
            datos_decodificados.append(datos)
        
        df_resultado = pd.DataFrame(datos_decodificados)
        df_final = pd.concat([df.reset_index(drop=True), df_resultado], axis=1)
        
        return df_final
    
    def procesar_todos_los_csv(self, patron="*.csv"):
        """Procesa todos los archivos CSV en el directorio actual"""
        archivos_csv = glob.glob(patron)
        
        if not archivos_csv:
            print("No se encontraron archivos CSV en el directorio actual")
            return None
        
        print(f"Encontrados {len(archivos_csv)} archivos CSV:")
        for archivo in archivos_csv:
            print(f"  - {archivo}")
        
        todos_los_datos = []
        for archivo in archivos_csv:
            resultado = self.procesar_archivo_csv(archivo)
            if resultado is not None:
                todos_los_datos.append(resultado)
        
        if todos_los_datos:
            df_completo = pd.concat(todos_los_datos, ignore_index=True)
            print(f"\n✓ Procesamiento completado. Total de registros: {len(df_completo)}")
            return df_completo
        else:
            print("✗ No se pudo procesar ningún archivo")
            return None
    
    def generar_reporte(self, df):
        """Genera un reporte resumen de las llamadas"""
        if df is None or len(df) == 0:
            print("No hay datos para generar reporte")
            return
        
        print("=" * 80)
        print("REPORTE COMPLETO DE LLAMADAS")
        print("=" * 80)
        print(f"Total de llamadas procesadas: {len(df):,}")
        
        # Manejar fechas con valores NaN
        if 'fecha' in df.columns:
            fechas_validas = df['fecha'].dropna()
            if len(fechas_validas) > 0:
                print(f"Período: {fechas_validas.min()} to {fechas_validas.max()}")
            else:
                print("Período: No hay fechas válidas")
        
        # Estadísticas por tipo de llamada
        if 'tipo_llamada' in df.columns:
            print("\n1. Distribución por tipo de llamada:")
            distribucion = df['tipo_llamada'].value_counts(dropna=False)
            total = len(df)
            for tipo, cantidad in distribucion.items():
                porcentaje = (cantidad / total * 100) if total > 0 else 0
                tipo_str = "Sin tipo" if pd.isna(tipo) else tipo
                print(f"   {tipo_str}: {cantidad:>6} ({porcentaje:.1f}%)")
        
        # Llamadas desde anonymous
        llamadas_anonymous = df[df['origen_numero'] == 'anonymous']
        if len(llamadas_anonymous) > 0:
            print(f"\n2. Llamadas anónimas: {len(llamadas_anonymous)}")
        
        # Top extensiones que más llaman
        if 'origen_nombre' in df.columns and 'tipo_origen' in df.columns:
            print("\n3. Top 10 extensiones que originan llamadas:")
            origenes_ext = df[df['tipo_origen'] == 'Extensión']
            if len(origenes_ext) > 0:
                origenes = origenes_ext['origen_nombre'].value_counts().head(10)
                for extension, cantidad in origenes.items():
                    print(f"   {extension}: {cantidad:>6}")
            else:
                print("   No hay llamadas desde extensiones")
        
        # Llamadas desde externos
        if 'origen_nombre' in df.columns and 'tipo_origen' in df.columns:
            print("\n4. Llamadas desde externos:")
            externos = df[df['tipo_origen'] == 'Externo']
            if len(externos) > 0:
                print(f"   Total: {len(externos)} llamadas externas")
                top_externos = externos['origen_numero'].value_counts().head(5)
                for externo, cantidad in top_externos.items():
                    print(f"   {externo}: {cantidad:>6}")
            else:
                print("   No hay llamadas desde externos")
        
        # Top destinos
        if 'destino_nombre' in df.columns:
            print("\n5. Top 10 destinos más llamados:")
            destinos = df['destino_nombre'].value_counts().head(10)
            for destino, cantidad in destinos.items():
                print(f"   {destino}: {cantidad:>6}")
        
        # Llamadas por fecha
        if 'fecha' in df.columns:
            print("\n6. Llamadas por fecha (top 10):")
            fechas = df['fecha'].value_counts().head(10)
            for fecha, cantidad in fechas.items():
                print(f"   {fecha}: {cantidad:>6}")
        
        # Distribución horaria
        if 'hora' in df.columns:
            print("\n7. Distribución por hora del día:")
            horas_validas = df['hora'].dropna()
            if len(horas_validas) > 0:
                horas_validas = horas_validas.str[:2]  # Solo la hora
                horas = horas_validas.value_counts().sort_index()
                for hora, cantidad in horas.items():
                    print(f"   {hora}:00 - {cantidad:>6} llamadas")
            else:
                print("   No hay horas válidas")
        
        # Archivos procesados
        if 'archivo_origen' in df.columns:
            print("\n8. Archivos procesados:")
            archivos = df['archivo_origen'].value_counts()
            for archivo, cantidad in archivos.items():
                print(f"   {archivo}: {cantidad:>6} registros")
        
        # Llamadas no decodificadas
        llamadas_no_decodificadas = df[df['tipo_llamada'] == 'No decodificada']
        if len(llamadas_no_decodificadas) > 0:
            print(f"\n9. Llamadas no decodificadas: {len(llamadas_no_decodificadas)}")
            print("   Ejemplos:")
            for i, archivo in enumerate(llamadas_no_decodificadas['nombre_archivo_original'].head(5)):
                print(f"   - {archivo}")

def main():
    """Función principal"""
    print("🔍 DECODIFICADOR DE LLAMADAS TRANSCRITAS")
    print("=" * 50)
    
    # Crear decodificador
    decodificador = DecodificadorLlamadas()
    
    # Procesar todos los CSV automáticamente
    df_resultado = decodificador.procesar_todos_los_csv(patron="*.csv")
    
    if df_resultado is not None:
        # Guardar resultado combinado
        archivo_salida = "llamadas_decodificadas_completas.csv"
        df_resultado.to_csv(archivo_salida, index=False, encoding='utf-8')
        print(f"\n💾 Resultado guardado en: {archivo_salida}")
        
        # Generar reporte
        print("\n" + "=" * 50)
        print("📊 GENERANDO REPORTE...")
        print("=" * 50)
        decodificador.generar_reporte(df_resultado)
        
        # Guardar reporte en archivo
        with open("reporte_llamadas.txt", "w", encoding='utf-8') as f:
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            decodificador.generar_reporte(df_resultado)
            sys.stdout = original_stdout
        print(f"\n📄 Reporte guardado en: reporte_llamadas.txt")
        
        # Mostrar preview
        print(f"\n👀 Vista previa de los datos:")
        columnas_preview = ['tipo_llamada', 'origen_nombre', 'destino_nombre', 'fecha', 'hora']
        columnas_disponibles = [col for col in columnas_preview if col in df_resultado.columns]
        if columnas_disponibles:
            print(df_resultado[columnas_disponibles].head(10))
        else:
            print("No hay columnas disponibles para mostrar")
    
    else:
        print("No se pudieron procesar los archivos CSV")

# Ejecutar automáticamente
if __name__ == "__main__":
    main()