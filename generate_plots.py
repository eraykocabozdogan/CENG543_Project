import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- AYARLAR ---
plt.style.use('default') 
COLOR_EM = '#2E86AB'    # Mavi
COLOR_GAP = '#D64045'   # Kırmızı
COLOR_BOX = '#ECF0F1'   # Gri Kutu
COLOR_PROXY = '#FFF8DC' # Krem (Proxy için)
COLOR_EDGE = '#2C3E50'  # Çerçeve

def draw_results_chart():
    """Tablo 3 verilerine göre Sonuç Grafiğini çizer."""
    print("Sonuç Grafiği çiziliyor...")
    
    strategies = ['Baseline', 'Placeholder', 'Faker']
    em_scores = [50.8, 15.2, 13.6]
    gaps = [-6.8, 50.0, 34.0]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, em_scores, width, label='Exact Match (EM)', color=COLOR_EM, edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, gaps, width, label='Hallucination Gap', color=COLOR_GAP, edgecolor='black', alpha=0.9)

    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Anonymization (Active Subset, N=250)', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linewidth=0.8)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            offset = 3 if height >= 0 else -15
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('fig1_results_chart.pdf')
    print(" -> fig1_results_chart.pdf oluşturuldu.")

def draw_architecture_diagram():
    """Proxy Mimarisini DÜZELTİLMİŞ hizalamalarla çizer."""
    print("Mimari Diyagramı çiziliyor...")
    
    # Figür boyutunu genişlettik ve margin verdik
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Koordinat Tanımları (Simetri için)
    y_center = 2.5
    
    # 1. User Box
    user_x, user_w = 0.5, 2.5
    user_h = 2.0
    user_y = y_center - user_h/2
    
    # 2. Proxy Box
    proxy_x, proxy_w = 4.5, 3.5  # Biraz daha geniş
    proxy_h = 2.8               # Biraz daha yüksek
    proxy_y = y_center - proxy_h/2
    
    # 3. RAG Box
    rag_x, rag_w = 9.5, 2.5
    rag_h = 2.0
    rag_y = y_center - rag_h/2

    def create_box(x, y, w, h, text, subtext, color=COLOR_BOX):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     linewidth=2, edgecolor=COLOR_EDGE, facecolor=color)
        ax.add_patch(box)
        # Ana Metin (Kutunun üst yarısına)
        ax.text(x + w/2, y + h*0.65, text, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=COLOR_EDGE)
        # Alt Metin
        if subtext:
            ax.text(x + w/2, y + h*0.45, subtext, ha='center', va='center', 
                    fontsize=9, style='italic', color='#555555')

    # Kutuları Çiz
    create_box(user_x, user_y, user_w, user_h, "User", "Query & Response")
    create_box(proxy_x, proxy_y, proxy_w, proxy_h, "PII Mapping\nProxy", "Stateful Translation Layer", COLOR_PROXY)
    create_box(rag_x, rag_y, rag_w, rag_h, "RAG / LLM", "Retrieval & Generation")

    # Proxy içine "Mapping Table" temsili ekle (Daha düzgün hizalı)
    table_box_y = proxy_y + 0.3
    table_box_h = 0.6
    table_box_w = 2.0
    table_box_x = proxy_x + (proxy_w - table_box_w)/2
    
    # Tablo kutusu (İç içe kutu efekti)
    table_rect = patches.Rectangle((table_box_x, table_box_y), table_box_w, table_box_h, 
                                   linewidth=1, edgecolor='#999', facecolor='white', alpha=0.8)
    ax.add_patch(table_rect)
    ax.text(table_box_x + table_box_w/2, table_box_y + table_box_h/2, 
            "{ Cam: C.Arm }", ha='center', va='center', 
            fontsize=8, fontfamily='monospace', color='#333')
    ax.text(table_box_x + table_box_w/2, table_box_y + table_box_h + 0.15, 
            "Transient State:", ha='center', va='bottom', fontsize=7, color='#666')


    # --- OKLAR ---
    # Üstten Gidiş (Query)
    arrow_y_top = y_center + 0.5
    # Alttan Dönüş (Response)
    arrow_y_bot = y_center - 0.5
    
    def draw_arrow(x1, x2, y, text_main, text_sub):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=COLOR_EDGE))
        # Yazılar
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + 0.15, text_main, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2980B9')
        ax.text(mid_x, y - 0.25, text_sub, ha='center', va='top', fontsize=8, color='#7F8C8D')

    # 1. User -> Proxy
    draw_arrow(user_x + user_w + 0.1, proxy_x - 0.1, arrow_y_top, 
               "1. Query", '"Who is Cam?"')
    
    # 2. Proxy -> RAG
    draw_arrow(proxy_x + proxy_w + 0.1, rag_x - 0.1, arrow_y_top, 
               "2. Anon.", '"Who is C.Arm?"')
    
    # 3. RAG -> Proxy
    draw_arrow(rag_x - 0.1, proxy_x + proxy_w + 0.1, arrow_y_bot, 
               "3. Hallucination", '"C.Arm is MVP..."')
    
    # 4. Proxy -> User
    draw_arrow(proxy_x - 0.1, user_x + user_w + 0.1, arrow_y_bot, 
               "4. De-anon.", '"Cam is MVP..."')

    # Tight layout kullanmadan manuel margin ile kaydet (Kaymayı önler)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig('fig2_architecture.pdf')
    print(" -> fig2_architecture.pdf DÜZELTİLMİŞ versiyonu oluşturuldu.")

if __name__ == "__main__":
    draw_results_chart()
    draw_architecture_diagram()