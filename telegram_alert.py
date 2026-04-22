import os
import requests
from src.data_fetcher import load_all_data
from src.signals import compute_log_zscore

def send_telegram_message(bot_token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Messaggio Telegram inviato con successo.")
    except Exception as e:
        print(f"Errore nell'invio del messaggio Telegram: {e}")

def main():
    # Recupera le credenziali dall'ambiente (passate da GitHub Actions)
    eodhd_key = os.environ.get("EODHD_API_KEY")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not all([eodhd_key, bot_token, chat_id]):
        print("Credenziali mancanti. Verifica i secrets in GitHub.")
        return

    # Parametri operativi (allineati ad app.py)
    window = 90
    upper_thresh = 2.5
    lower_thresh = -2.0

    print("Scaricamento dati EODHD...")
    df = load_all_data(eodhd_key)
    
    if df.empty or len(df) < window:
        print("Dati insufficienti o vuoti.")
        return

    # Calcolo Z-Score
    zscore = compute_log_zscore(df["vvix"], window=window)
    
    curr_vvix = df["vvix"].iloc[-1]
    curr_vix  = df["vix"].iloc[-1]
    curr_spx  = df["spx"].iloc[-1]
    curr_z    = zscore.dropna().iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")

    # Verifica Regole
    rule1 = (curr_z >= upper_thresh) and (15 <= curr_vix < 20)
    rule3 = (curr_z >= upper_thresh) and (curr_vix >= 20)
    rule2 = (curr_z <= lower_thresh) and (curr_vix < 15)
    rule4 = (curr_z <= lower_thresh) and (15 <= curr_vix < 20)

    signal_text = ""
    if rule1:
        signal_text = "🔴 <b>REGOLA 1 ATTIVA: Short Volatilità</b>\nMean reversion della vola in regime normale."
    elif rule3:
        signal_text = "🩸 <b>REGOLA 3 ATTIVA: Blow-off Top</b>\nEsaurimento panico. Setup di short massiccio IV."
    elif rule2:
        signal_text = "🟢 <b>REGOLA 2 ATTIVA: Cautela / Complacenza</b>\nRischio latente. Valutare coperture."
    elif rule4:
        signal_text = "🟡 <b>REGOLA 4 ATTIVA: VIX Pop</b>\nQuiete pre-tempesta. Possibile spike VIX a 5-15 gg."

    # COSTRUZIONE DEL MESSAGGIO (viene costruito sempre, in ogni caso)
    if signal_text:
        # Messaggio se c'è un Allarme
        message = (
            f"🔔 <b>Kriterion Quant - VVIX Alert</b> 🔔\n\n"
            f"Data: {last_date}\n"
            f"{signal_text}\n\n"
            f"📊 <b>Dati di chiusura:</b>\n"
            f"• VVIX: {curr_vvix:.2f}\n"
            f"• VIX: {curr_vix:.2f}\n"
            f"• S&P 500: {curr_spx:.2f}\n"
            f"• Log-zScore: <b>{curr_z:.3f}</b>\n\n"
            f"<i>Soglie: OB {upper_thresh} | OS {lower_thresh} | Finestra: {window}gg</i>"
        )
    else:
        # Messaggio "Daily Recap" inviato nei giorni normali senza setup
        print(f"Nessuna regola attiva ({last_date}). Z-Score: {curr_z:.3f}, VIX: {curr_vix:.2f}.")
        message = (
            f"ℹ️ <b>Kriterion Quant - Daily Recap</b>\n\n"
            f"Data: {last_date}\n"
            f"Nessun setup VVIX estremo registrato oggi.\n\n"
            f"📊 <b>Dati di chiusura:</b>\n"
            f"• VVIX: {curr_vvix:.2f}\n"
            f"• VIX: {curr_vix:.2f}\n"
            f"• S&P 500: {curr_spx:.2f}\n"
            f"• Log-zScore: <b>{curr_z:.3f}</b>\n\n"
            f"<i>(Mercato in regime neutro)</i>"
        )

    # INVIO SU TELEGRAM (eseguito SEMPRE, sia che ci sia un alert sia che ci sia il recap)
    send_telegram_message(bot_token, chat_id, message)

if __name__ == "__main__":
    main()
