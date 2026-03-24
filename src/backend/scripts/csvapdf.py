import csv
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def csv_to_pdf(input_csv, output_pdf):
    doc = SimpleDocTemplate(output_pdf, pagesize=landscape(A4))
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"Reporte: {os.path.basename(input_csv)}", styles["Title"]))

    data = []
    try:
        # Usamos delimitador tabulador (\t) porque es lo estándar en tus archivos anteriores
        with open(input_csv, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Truncar para que quepa en el PDF
                clean_row = [item[:50] + "..." if len(item) > 50 else item for item in row]
                data.append(clean_row)

        table = Table(data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]
            )
        )
        elements.append(table)
        doc.build(elements)
        print(f"PDF guardado en: {os.path.abspath(output_pdf)}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error: {e}")


if __name__ == "__main__":
    # RUTA CORREGIDA SEGÚN TU IMAGEN:
    # 1. Obtenemos la carpeta donde está este script (lab3_rag)
    base_path = os.path.dirname(__file__)
    # 2. Subimos un nivel (..) y entramos en la carpeta 'data'
    archivo_entrada = os.path.abspath(
        os.path.join(base_path, "..", "..", "data", "training", "training.csv")
    )
    archivo_salida = os.path.abspath(
        os.path.join(base_path, "..", "..", "evaluation", "results", "reporte_training.pdf")
    )

    if os.path.exists(archivo_entrada):
        csv_to_pdf(archivo_entrada, archivo_salida)
    else:
        print(f"No se encuentra el archivo en: {os.path.abspath(archivo_entrada)}")
