# generate_faqs.py - La nostra fabbrica di pagine SEO

import json
from datetime import datetime, date
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def create_faq_pages():
    """
    Legge le domande da faq_data.json, crea una pagina HTML per ciascuna
    e aggiorna la sitemap.
    """
    print("--- Inizio generazione pagine FAQ ---")

    # URL base del sito per link e sitemap
    base_url = "https://davidoggoo.github.io/Finance_Machine_Learning"

    # Carica le domande e risposte
    with open('faq_data.json', 'r', encoding='utf-8') as f:
        faqs = json.load(f)

    # Assicura che la cartella /faq esista
    if not os.path.exists('faq'):
        os.makedirs('faq')

    # Template HTML per ogni pagina FAQ
    html_template = """<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{question}</title>
    <meta name="description" content="{description}">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f8f9fa; color: #333; }}
        header {{ background-color: #fff; padding: 1rem; border-bottom: 1px solid #ddd; text-align: center; }}
        nav a {{ text-decoration: none; color: #333; margin: 0 15px; font-weight: 500; }}
        main {{ max-width: 800px; margin: 2rem auto; padding: 1rem; background: #fff; border-radius: 8px; }}
        a {{ color: #007bff; }}
    </style>
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "FAQPage",
      "mainEntity": [{{
        "@type": "Question",
        "name": "{question}",
        "acceptedAnswer": {{
          "@type": "Answer",
          "text": "{answer_text}"
        }}
      }}]
    }}
    </script>
</head>
<body>
    <header>
        <nav>
          <a href="/Finance_Machine_Learning/">Forecast S&P 500</a>
          <a href="/Finance_Machine_Learning/mood-dial.html">Mood Trader Dial</a>
          <a href="/Finance_Machine_Learning/roulette.html">AI Trade Roulette</a>
          <a href="/Finance_Machine_Learning/quiz.html">Flash Quiz</a>
          <a href="/Finance_Machine_Learning/gallery.html">Galleria Grafici</a>
        </nav>
    </header>
    <main>
        <h1>{question}</h1>
        {answer_html}
        <hr>
        <p><a href="/Finance_Machine_Learning/">&larr; Torna alla Home</a></p>
    </main>
</body>
</html>"""

    sitemap_urls = []
    # Aggiungi le pagine esistenti alla lista per la sitemap
    for page in ["", "mood-dial.html", "roulette.html", "quiz.html", "gallery.html"]:
        sitemap_urls.append(f"{base_url}/{page}")

    for faq in faqs:
        # Crea la pagina HTML
        page_content = html_template.format(
            question=faq['question'],
            description=faq['answer_html'].split('</p>')[0].replace('<p>', ''),
            answer_html=faq['answer_html'],
            answer_text=faq['answer_html'].replace('"', '\\"').replace('\n', ' ').replace('<p>', '').replace('</p>','').replace('<ul>','').replace('</ul>','').replace('<li>','').replace('</li>','').replace('<strong>','').replace('</strong>','')
        )
        file_path = os.path.join('faq', f"{faq['slug']}.html")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(page_content)
        print(f"Pagina creata: {file_path}")
        
        # Aggiungi l'URL alla lista per la sitemap
        sitemap_urls.append(f"{base_url}/faq/{faq['slug']}.html")

    # Aggiorna la sitemap.xml
    update_sitemap(sitemap_urls)


def update_sitemap(url_list):
    """
    Genera un nuovo sitemap.xml con tutti gli URL del sito.
    """
    print("--- Aggiornamento sitemap.xml ---")
    urlset = Element('urlset', xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    
    for url in url_list:
        url_element = SubElement(urlset, 'url')
        loc = SubElement(url_element, 'loc')
        loc.text = url
        lastmod = SubElement(url_element, 'lastmod')
        lastmod.text = date.today().strftime('%Y-%m-%d')

    # Formatta l'XML per essere leggibile
    xml_str = parseString(tostring(urlset)).toprettyxml(indent="  ")
    
    with open('sitemap.xml', 'w', encoding='utf-8') as f:
        f.write(xml_str)
    print("File 'sitemap.xml' aggiornato con successo.")


if __name__ == "__main__":
    create_faq_pages()
