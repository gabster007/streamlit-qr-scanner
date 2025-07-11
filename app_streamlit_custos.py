import os
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from io import BytesIO
import pandas as pd
import re
from openai import OpenAI
import sqlite3
from datetime import datetime
import uuid
import cv2
from pyzbar.pyzbar import decode
import requests
from bs4 import BeautifulSoup
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Carrega a chave dos secrets do Streamlit
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect("invoices.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            cnpj TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id TEXT PRIMARY KEY,
            company_id TEXT,
            upload_date TEXT,
            empresa TEXT,
            cnpj TEXT,
            endereco TEXT,
            doc_type TEXT,
            FOREIGN KEY (company_id) REFERENCES companies (id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS invoice_items (
            id TEXT PRIMARY KEY,
            invoice_id TEXT,
            produto TEXT,
            codigo TEXT,
            quantidade REAL,
            unidade TEXT,
            valor_unitario REAL,
            valor_total REAL,
            classificacao TEXT,
            FOREIGN KEY (invoice_id) REFERENCES invoices (id)
        )
    """)
    conn.commit()
    conn.close()

# --- ReaderAgent: extrai texto bruto, com fallback OCR ---
class ReaderAgent:
    def run(self, pdf_bytes: bytes) -> str:
        text = ""
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        if len(text.strip()) < 100:
            try:
                images = convert_from_bytes(pdf_bytes)
                for img in images:
                    text += pytesseract.image_to_string(img, lang='por') + "\n"
            except Exception as e:
                st.warning(f"⚠️ Falha no OCR: {str(e)}")
        return text

# --- ExtractorAgent: identifica empresa, CNPJ, endereço e itens ---
class ExtractorAgent:
    def run(self, text: str) -> dict:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        empresa = cnpj = endereco = ""
        itens = []
        is_energy_bill = "ENERGIA ELÉTRICA" in text.upper()

        if is_energy_bill:
            for i, line in enumerate(lines):
                if "CNPJ" in line.upper():
                    cnpj_match = re.search(r"CNPJ[\s:]*([\d\.\-/]+)", line, re.IGNORECASE)
                    if cnpj_match:
                        cnpj = cnpj_match.group(1).strip()
                    empresa = lines[i-1].strip() if i > 0 else ""
                    endereco = lines[i+1].strip() if i+1 < len(lines) else ""
                    break
            return {"empresa": empresa, "cnpj": cnpj, "endereco": endereco, "itens_df": pd.DataFrame(), "doc_type": "energy_bill"}

        for i, line in enumerate(lines):
            if "CNPJ" in line.upper():
                cnpj_match = re.search(r"CNPJ[\s:]*([\d\.\-/]+)", line, re.IGNORECASE)
                if cnpj_match:
                    cnpj = cnpj_match.group(1).strip()
                empresa = lines[i-1].strip() if i > 0 else ""
                endereco = lines[i+1].strip() if i+1 < len(lines) else ""
                break

        for i, line in enumerate(lines):
            if "(Código:" in line or re.search(r"\(\s*Código\s*:\s*\d+\s*\)", line, re.IGNORECASE):
                code_match = re.search(r"\(\s*Código\s*:\s*(\d+)\s*\)", line, re.IGNORECASE)
                codigo = code_match.group(1) if code_match else ""
                prefix = re.split(r"\(\s*Código\s*:", line, flags=re.IGNORECASE)[0].strip()
                prefix = prefix.replace("Vl. Total", "").replace("Vlr. Total", "").strip()
                words = prefix.split()
                if len(words) < 3 and i > 0 and any(x in lines[i-1].lower() for x in ["vl. total", "vlr. total"]):
                    prev = lines[i-1].split("Vl. Total")[0].split("Vlr. Total")[0].strip()
                    nome = f"{prev} {prefix}".strip()
                else:
                    nome = prefix

                qtde = None
                unidade = None
                unitario = None
                total = None

                for j in range(i, min(len(lines), i+10)):
                    l = lines[j].strip()
                    qtde_match = re.search(r"Qtde\.?\s*[:=]?\s*([\d,\.]+)", l, re.IGNORECASE)
                    unit_match = re.search(r"UN\.?\s*[:=]?\s*(\w+)", l, re.IGNORECASE)
                    unitario_match = re.search(r"Vl\.?\s*Unit\.?\s*[:=]?\s*([\d,\.]+)", l, re.IGNORECASE)
                    total_match = re.search(r"Vl\.?\s*Total\s*[:=]?\s*([\d,\.]+)", l, re.IGNORECASE)
                    if qtde_match:
                        try:
                            qtde = float(qtde_match.group(1).replace(',', '.'))
                        except:
                            qtde = None
                    if unit_match:
                        unidade = unit_match.group(1)
                    if unitario_match:
                        try:
                            unitario = float(unitario_match.group(1).replace(',', '.'))
                        except:
                            unitario = None
                    if total_match:
                        try:
                            total = float(total_match.group(1).replace(',', '.'))
                        except:
                            total = None
                    if total is None and re.fullmatch(r"[\d,\.]+", l):
                        try:
                            total = float(l.replace(',', '.'))
                        except:
                            total = None
                    if qtde and unidade and unitario and total:
                        break

                if nome and (qtde or total):
                    itens.append({
                        "Produto": nome,
                        "Código": codigo,
                        "Quantidade": qtde,
                        "Unidade": unidade,
                        "Valor Unitário (R$)": unitario,
                        "Valor Total (R$)": total
                    })

        df_itens = pd.DataFrame(itens)
        return {"empresa": empresa, "cnpj": cnpj, "endereco": endereco, "itens_df": df_itens, "doc_type": "nfce"}

# --- ClassifierAgent: classifica cada item usando OpenAI ---
class ClassifierAgent:
    def run(self, itens_df: pd.DataFrame) -> pd.DataFrame:
        if itens_df.empty or "Produto" not in itens_df.columns:
            itens_df["Classificação"] = []
            return itens_df

        def classify(name: str) -> str:
            if not name or not isinstance(name, str) or len(name.strip()) < 3:
                return "Variável"
            try:
                prompt = f"""
                Você é um assistente especializado em contabilidade e análise de notas fiscais brasileiras.
                Classifique o seguinte item de uma nota fiscal em uma das categorias: 'Custo', 'Despesa' ou 'Variável'.
                - 'Custo': Itens diretamente relacionados à produção ou bens essenciais (ex.: leite, carne, legumes).
                - 'Despesa': Itens relacionados a manutenção ou consumo não produtivo (ex.: papel higiênico, detergente).
                - 'Variável': Outros itens que não se encaixam claramente nas categorias acima.
                Item: {name}
                Forneça apenas a categoria ('Custo', 'Despesa' ou 'Variável') como resposta.
                """
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Você é um assistente de classificação contábil."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                st.warning(f"⚠️ Erro ao classificar '{name}': {str(e)}. Usando classificação padrão.")
                return "Variável"

        itens_df["Classificação"] = itens_df["Produto"].apply(classify)
        return itens_df

# --- QRScannerAgent: extrai dados do QR code ---
class QRScannerAgent:
    def run(self, image) -> dict:
        # Decode QR code from image frame
        decoded_objects = decode(image)
        if not decoded_objects:
            return {"empresa": "", "cnpj": "", "endereco": "", "itens_df": pd.DataFrame(), "doc_type": "unknown"}

        qr_data = decoded_objects[0].data.decode("utf-8")
        st.success(f"✅ QR code detectado: {qr_data}")

        # Assume QR code contains NFC-e URL
        if qr_data.startswith("http"):
            try:
                response = requests.get(qr_data, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator="\n")
            except Exception as e:
                st.error(f"❌ Erro ao acessar URL do QR code: {str(e)}")
                return {"empresa": "", "cnpj": "", "endereco": "", "itens_df": pd.DataFrame(), "doc_type": "unknown"}
        else:
            text = qr_data

        # Use ExtractorAgent to parse the text
        return ExtractorAgent().run(text)

# --- VideoTransformer for Webcam ---
class QRVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.result = None

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        result = QRScannerAgent().run(image)
        if result["itens_df"].empty and "cnpj" not in result or not result["cnpj"]:
            return image  # Continue scanning if no valid data
        self.result = result
        return image  # Stop or continue based on your preference

# --- Database Functions ---
def add_company(name: str, cnpj: str = ""):
    conn = sqlite3.connect("invoices.db")
    c = conn.cursor()
    company_id = str(uuid.uuid4())
    c.execute("INSERT INTO companies (id, name, cnpj) VALUES (?, ?, ?)", (company_id, name, cnpj))
    conn.commit()
    conn.close()
    return company_id

def get_companies():
    conn = sqlite3.connect("invoices.db")
    c = conn.cursor()
    c.execute("SELECT id, name, cnpj FROM companies")
    companies = c.fetchall()
    conn.close()
    return companies

def save_invoice(company_id: str, result: dict, file_bytes: bytes = None):
    conn = sqlite3.connect("invoices.db")
    c = conn.cursor()
    invoice_id = str(uuid.uuid4())
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO invoices (id, company_id, upload_date, empresa, cnpj, endereco, doc_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (invoice_id, company_id, upload_date, result["empresa"], result["cnpj"], result["endereco"], result["doc_type"]))
    
    for _, row in result["itens_df"].iterrows():
        item_id = str(uuid.uuid4())
        c.execute("""
            INSERT INTO invoice_items (id, invoice_id, produto, codigo, quantidade, unidade, valor_unitario, valor_total, classificacao)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id, invoice_id, row["Produto"], row["Código"], row["Quantidade"], row["Unidade"],
            row["Valor Unitário (R$)"], row["Valor Total (R$)"], row.get("Classificação", "Variável")
        ))
    
    conn.commit()
    conn.close()
    return invoice_id

def get_invoices(company_id: str):
    conn = sqlite3.connect("invoices.db")
    c = conn.cursor()
    c.execute("SELECT id, upload_date, empresa, cnpj, endereco, doc_type FROM invoices WHERE company_id = ?", (company_id,))
    invoices = c.fetchall()
    conn.close()
    return invoices

def get_invoice_items(invoice_id: str):
    conn = sqlite3.connect("invoices.db")
    df = pd.read_sql_query("""
        SELECT produto AS Produto, codigo AS Código, quantidade AS Quantidade, unidade AS Unidade,
               valor_unitario AS "Valor Unitário (R$)", valor_total AS "Valor Total (R$)", classificacao AS Classificação
        FROM invoice_items WHERE invoice_id = ?
    """, conn, params=(invoice_id,))
    conn.close()
    return df

def update_invoice_item(item_id: str, updates: dict):
    conn = sqlite3.connect("invoices.db")
    c = conn.cursor()
    c.execute("""
        UPDATE invoice_items SET
            produto = ?, codigo = ?, quantidade = ?, unidade = ?, valor_unitario = ?, valor_total = ?, classificacao = ?
        WHERE id = ?
    """, (
        updates.get("Produto"), updates.get("Código"), updates.get("Quantidade"), updates.get("Unidade"),
        updates.get("Valor Unitário (R$)"), updates.get("Valor Total (R$)"), updates.get("Classificação"), item_id
    ))
    conn.commit()
    conn.close()

def delete_invoice(invoice_id: str):
    conn = sqlite3.connect("invoices.db")
    c = conn.cursor()
    c.execute("DELETE FROM invoice_items WHERE invoice_id = ?", (invoice_id,))
    c.execute("DELETE FROM invoices WHERE id = ?", (invoice_id,))
    conn.commit()
    conn.close()

def generate_monthly_report(company_id: str, year: int, month: int):
    conn = sqlite3.connect("invoices.db")
    query = """
        SELECT i.upload_date, i.empresa, i.cnpj, i.endereco, i.doc_type, ii.produto, ii.codigo, ii.quantidade,
               ii.unidade, ii.valor_unitario, ii.valor_total, ii.classificacao
        FROM invoices i
        JOIN invoice_items ii ON i.id = ii.invoice_id
        WHERE i.company_id = ? AND strftime('%Y', i.upload_date) = ? AND strftime('%m', i.upload_date) = ?
    """
    df = pd.read_sql_query(query, conn, params=(company_id, str(year), f"{month:02d}"))
    conn.close()
    
    if df.empty:
        return None, None
    
    summary = df.groupby("classificacao").agg({
        "valor_total": "sum",
        "quantidade": "sum"
    }).reset_index()
    summary.columns = ["Classificação", "Valor Total (R$)", "Quantidade Total"]
    
    return df, summary

# --- Pipeline manual ---
def executar_pipeline(file_bytes: bytes, is_pdf: bool = True) -> dict:
    if is_pdf:
        return executar_pipeline_pdf(file_bytes)
    else:
        return QRScannerAgent().run(file_bytes)

def executar_pipeline_pdf(pdf_bytes: bytes) -> dict:
    raw_text = ReaderAgent().run(pdf_bytes)
    data = ExtractorAgent().run(raw_text)
    df = data["itens_df"]
    if not df.empty:
        data["itens_df"] = ClassifierAgent().run(df)
    return data

# --- Streamlit UI ---
st.set_page_config(page_title="Classificador de Custos - NFC-e", layout="wide")
st.title("📊 Classificador de Gastos/Despesas via Nota Fiscal")

# Initialize database
init_db()

# Verifica se a chave da API está configurada
if not os.environ["OPENAI_API_KEY"]:
    st.error("❌ Chave da API OpenAI não configurada. Adicione 'OPENAI_API_KEY' nos secrets do Streamlit.")
else:
    # Sidebar for company management
    st.sidebar.header("Gerenciar Empresas")
    company_name = st.sidebar.text_input("Nome da Empresa")
    company_cnpj = st.sidebar.text_input("CNPJ (opcional)")
    if st.sidebar.button("Adicionar Empresa"):
        if company_name:
            add_company(company_name, company_cnpj)
            st.sidebar.success(f"✅ Empresa '{company_name}' adicionada!")
        else:
            st.sidebar.error("❌ Informe o nome da empresa.")

    # Select company
    companies = get_companies()
    company_options = {c[1]: c[0] for c in companies}
    selected_company = st.sidebar.selectbox("Selecionar Empresa", options=[""] + list(company_options.keys()))
    selected_company_id = company_options.get(selected_company)

    # Main content
    if selected_company:
        st.header(f"Gerenciar Notas Fiscais - {selected_company}")
        
        # Upload file (PDF or Image with QR)
        st.subheader("Carregar Arquivo")
        uploaded_file = st.file_uploader("📁 Envie o PDF da Nota Fiscal ou Imagem com QR Code", type=["pdf", "jpg", "jpeg", "png"], key=f"uploader_{selected_company}")
        if uploaded_file:
            with st.spinner("🔍 Processando arquivo..."):
                try:
                    is_pdf = uploaded_file.type == "application/pdf"
                    result = executar_pipeline(uploaded_file.read(), is_pdf)
                    df = result["itens_df"]
                    doc_type = result.get("doc_type", "unknown")
                    invoice_id = save_invoice(selected_company_id, result, uploaded_file.read())
                    st.success(f"✅ Nota fiscal processada e salva com ID {invoice_id}!")
                    st.markdown(f"**Empresa:** {result['empresa']}")
                    st.markdown(f"**CNPJ:** {result['cnpj']}")
                    st.markdown(f"**Endereço:** {result['endereco']}")
                    if doc_type == "energy_bill":
                        st.warning("⚠️ Este é um documento de energia elétrica. Apenas informações básicas foram extraídas.")
                    elif df.empty:
                        st.warning("⚠️ Nenhum item encontrado. Verifique o layout do arquivo ou se é uma NFC-e válida.")
                    else:
                        st.markdown("### 🧾 Itens Encontrados e Classificados")
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "⬇️ Baixar CSV",
                            df.to_csv(index=False).encode('utf-8'),
                            "itens_classificados.csv",
                            "text/csv"
                        )
                except Exception as e:
                    st.error(f"❌ Erro ao processar o arquivo: {str(e)}")

        # Real-time QR Code Scanning with Camera
        st.subheader("Escanear QR Code com Câmera")
        qr_transformer = QRVideoTransformer()
        webrtc_streamer(
            key="qr-scanner",
            video_transformer_factory=lambda: qr_transformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )

        if qr_transformer.result and not qr_transformer.result["itens_df"].empty:
            result = qr_transformer.result
            invoice_id = save_invoice(selected_company_id, result)
            st.success(f"✅ Nota fiscal escaneada e salva com ID {invoice_id}!")
            st.markdown(f"**Empresa:** {result['empresa']}")
            st.markdown(f"**CNPJ:** {result['cnpj']}")
            st.markdown(f"**Endereço:** {result['endereco']}")
            if result["doc_type"] == "energy_bill":
                st.warning("⚠️ Este é um documento de energia elétrica. Apenas informações básicas foram extraídas.")
            elif not result["itens_df"].empty:
                st.markdown("### 🧾 Itens Encontrados e Classificados")
                st.dataframe(result["itens_df"], use_container_width=True)
                st.download_button(
                    "⬇️ Baixar CSV",
                    result["itens_df"].to_csv(index=False).encode('utf-8'),
                    "itens_classificados.csv",
                    "text/csv"
                )
            qr_transformer.result = None  # Reset after processing

        # Display saved invoices
        st.subheader("Notas Fiscais Salvas")
        invoices = get_invoices(selected_company_id)
        if invoices:
            for invoice in invoices:
                invoice_id, upload_date, empresa, cnpj, endereco, doc_type = invoice
                with st.expander(f"Nota {invoice_id} - {upload_date}"):
                    st.markdown(f"**Empresa:** {empresa}")
                    st.markdown(f"**CNPJ:** {cnpj}")
                    st.markdown(f"**Endereço:** {endereco}")
                    st.markdown(f"**Tipo:** {'Energia Elétrica' if doc_type == 'energy_bill' else 'NFC-e'}")
                    df_items = get_invoice_items(invoice_id)
                    if not df_items.empty:
                        st.dataframe(df_items, use_container_width=True)
                        st.subheader("Editar Itens")
                        for idx, row in df_items.iterrows():
                            with st.form(f"edit_item_{row['Produto']}_{idx}"):
                                edited_row = {
                                    "Produto": st.text_input("Produto", row["Produto"], key=f"produto_{invoice_id}_{idx}"),
                                    "Código": st.text_input("Código", row["Código"], key=f"codigo_{invoice_id}_{idx}"),
                                    "Quantidade": st.number_input("Quantidade", value=row["Quantidade"] or 0.0, step=0.01, key=f"qtde_{invoice_id}_{idx}"),
                                    "Unidade": st.text_input("Unidade", row["Unidade"], key=f"unidade_{invoice_id}_{idx}"),
                                    "Valor Unitário (R$)": st.number_input("Valor Unitário (R$)", value=row["Valor Unitário (R$)"] or 0.0, step=0.01, key=f"unitario_{invoice_id}_{idx}"),
                                    "Valor Total (R$)": st.number_input("Valor Total (R$)", value=row["Valor Total (R$)"] or 0.0, step=0.01, key=f"total_{invoice_id}_{idx}"),
                                    "Classificação": st.selectbox("Classificação", ["Custo", "Despesa", "Variável"], index=["Custo", "Despesa", "Variável"].index(row["Classificação"]), key=f"class_{invoice_id}_{idx}")
                                }
                                if st.form_submit_button("Salvar Alterações"):
                                    update_invoice_item(f"{invoice_id}_{idx}", edited_row)
                                    st.success(f"✅ Item {row['Produto']} atualizado!")
                    if st.button("🗑️ Apagar Nota", key=f"delete_{invoice_id}"):
                        delete_invoice(invoice_id)
                        st.success(f"✅ Nota {invoice_id} apagada!")
        else:
            st.info("📌 Nenhuma nota fiscal salva para esta empresa.")

        # Monthly Report
        st.subheader("Relatório Mensal")
        year = st.number_input("Ano", min_value=2000, max_value=2030, value=datetime.now().year)
        month = st.number_input("Mês", min_value=1, max_value=12, value=datetime.now().month)
        if st.button("Gerar Relatório"):
            df_report, summary = generate_monthly_report(selected_company_id, year, month)
            if df_report is not None:
                st.markdown(f"### Relatório de {month}/{year} para {selected_company}")
                st.markdown("#### Resumo por Classificação")
                st.dataframe(summary, use_container_width=True)
                st.markdown("#### Detalhes das Notas")
                st.dataframe(df_report, use_container_width=True)
                csv = df_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Baixar Relatório CSV",
                    csv,
                    f"relatorio_{selected_company}_{year}_{month:02d}.csv",
                    "text/csv"
                )
            else:
                st.warning(f"⚠️ Nenhuma nota encontrada para {month}/{year}.")
    else:
        st.info("📌 Selecione ou adicione uma empresa para gerenciar notas fiscais.")
