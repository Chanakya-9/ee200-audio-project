import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
import time
import pandas as pd
from collections import defaultdict
from scipy.ndimage import maximum_filter
import back



st.set_page_config(page_title="EE200 Project", layout="wide")

st.title("EE200: Audio Fingerprinting Project")
st.write("Signals, Systems & Networks - Project Code Framework")

tab1, tab2, tab3 = st.tabs(["Library Database", "Identify Track", "Batch Processing"])

def save_temp_file(file):
    suffix = ".mp3" if file.name.endswith(".mp3") else ".wav"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(file.read())
    temp.close()
    return temp.name

with tab1:
    st.header("Songs in Database")
    if not back.database:
        st.write("Database is empty. Add .mp3 files to songs folder.")
    else:
        cols = st.columns(5)
        for idx, (song_name, hashes) in enumerate(back.database.items()):
            col_idx = idx % 5
            with cols[col_idx]:
                with st.container(border=True):
                    clean_name = song_name[:-4] if song_name.lower().endswith(('.mp3', '.wav')) else song_name
                    st.markdown(f"**{clean_name}**")
                    st.caption(f"{len(hashes):,} hashes")
                    
                    
                    fig, ax = plt.subplots(figsize=(2, 1.2))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    if len(hashes) > 0:
                        t1_vals = [h[3] for h in hashes[:250]]
                        f1_vals = [h[0] for h in hashes[:250]]
                        
                        ax.scatter(t1_vals, f1_vals, s=0.2, c='#39d353', alpha=0.6)
                        
                    ax.axis('off')
                    plt.tight_layout(pad=0)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

with tab2:
    st.header("🎯 Identify a Sample Clip")
    st.write("Upload an audio query snippet to test the identifier.")
    
    if "processed_file_id" not in st.session_state:
        st.session_state.processed_file_id = None

    uploaded_file = st.file_uploader(
        "Upload audio track file (.mp3 or .wav)", 
        type=["mp3", "wav"],
        key="audio_fingerprint_uploader"
    )
    
    output_canvas = st.empty()

    if uploaded_file is not None:
        if st.session_state.processed_file_id != uploaded_file.name:
            output_canvas.empty()
            st.session_state.processed_file_id = uploaded_file.name
            st.rerun()  

        with output_canvas.container():
            with st.spinner("Analyzing audio fingerprints..."):
                path = save_temp_file(uploaded_file)
                    
                t_start = time.time()
                query_hashes, S_db, sr_q, peak_coords = back.extract_features(path)
                t_features = int((time.time() - t_start) * 1000)
                
                t_start = time.time()
                votes = defaultdict(lambda: defaultdict(int))
                for q in query_hashes:
                    matches = back.index.get((q[0], q[1], q[2]))
                    if matches:
                        for song, t_song in matches:
                            offset = t_song - q[3]
                            votes[song][offset] += 1
                t_lookup = int((time.time() - t_start) * 1000)
                
                total_time = t_features + t_lookup

            if not votes:
                st.error("No matching tracks found in index.")
            else:
                best_song = max(votes, key=lambda s: max(votes[s].values()))
                max_agreement = max(votes[best_song].values())
                
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1: st.metric(label="⏱️ Feature Extraction", value=f"{t_features} ms")
                with b_col2: st.metric(label="🔍 DB Lookup", value=f"{t_lookup} ms")
                with b_col3: st.metric(label="⚡ Total Time", value=f"{total_time} ms")
                    
                st.write("---")
                
                clean_predicted_name = best_song[:-4] if best_song.endswith('.mp3') else best_song
                st.success(f"### MATCH FOUND: **{clean_predicted_name}**")
                st.caption(f"Confidence Identity Peak: {max_agreement} aligned cluster votes | Total Query Hashes analyzed: {len(query_hashes):,}")
                
                st.subheader("📋 Candidate Match Standings (Top 5)")
                candidate_list = []
                for song in votes:
                    max_votes = max(votes[song].values())
                    candidate_list.append({
                        "Track Name": song[:-4] if song.endswith('.mp3') else song, 
                        "Highest Alignment Peak Score": max_votes
                    })
                
                df_candidates = pd.DataFrame(candidate_list).sort_values(by="Highest Alignment Peak Score", ascending=False).head(5)
                st.dataframe(df_candidates, use_container_width=True, hide_index=True)
                
                st.write("---")
                
                st.subheader("Step 1: Feature Extraction Analysis")
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    with st.container(border=True):
                        st.markdown("**From Audio Signal to Spectrogram**")
                        fig1, ax1 = plt.subplots(figsize=(6, 4.2))
                        fig1.patch.set_facecolor('none')
                        ax1.set_facecolor('#0d1117')
                        
                        img = librosa.display.specshow(S_db, sr=sr_q, x_axis="time", y_axis="linear", cmap='magma', ax=ax1)
                        cbar = plt.colorbar(img, ax=ax1, format="%+2.0f dB")
                        cbar.ax.tick_params(labelsize=10, colors='white')
                        
                        ax1.tick_params(colors='white')
                        ax1.xaxis.label.set_color('white')
                        ax1.yaxis.label.set_color('white')
                        plt.tight_layout()
                        st.pyplot(fig1, use_container_width=True)
                    
                with col_p2:
                    with st.container(border=True):
                        st.markdown("**From Spectrogram to Constellation Landmarks**")
                        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
                        fig2.patch.set_facecolor('none')
                        ax2.set_facecolor('#0d1117')
                        
                        if len(peak_coords) > 0:
                            ax2.scatter(peak_coords[:, 1], peak_coords[:, 0], s=8, c='cyan', label='Peaks')
                            padding = 5
                            ax2.set_ylim(peak_coords[:, 0].min() - padding, peak_coords[:, 0].max() + padding)
                            ax2.set_xlim(peak_coords[:, 1].min() - padding, peak_coords[:, 1].max() + padding)
                            
                        ax2.set_xlabel("Time Frames")
                        ax2.set_ylabel("Frequency Bins")
                        ax2.tick_params(colors='white')
                        ax2.xaxis.label.set_color('white')
                        ax2.yaxis.label.set_color('white')
                        plt.tight_layout()
                        st.pyplot(fig2, use_container_width=True)
                        
                st.write("---")
                st.subheader("Step 2: Database Search & Coherence Verification")
                with st.container(border=True):
                    st.markdown("**The Alignment Spike (Time-Offset Histogram Verification)**")
                    fig3, ax3 = plt.subplots(figsize=(11, 3.8))
                    fig3.patch.set_facecolor('none')
                    ax3.set_facecolor('#0d1117')
                    
                    best_offsets = []
                    for off, count in votes[best_song].items():
                        best_offsets.extend([off] * count)
                        
                    ax3.hist(best_offsets, bins=60, color='#ff7b72', edgecolor='black', rwidth=0.85)
                    ax3.set_xlabel("Time Offset (Database Track Frame - Query Track Frame)")
                    ax3.set_ylabel("Fingerprint Pair Agreement Count")
                    ax3.grid(axis='y', linestyle='--', color='#30363d', alpha=0.5)
                    ax3.tick_params(colors='white')
                    ax3.xaxis.label.set_color('white')
                    ax3.yaxis.label.set_color('white')
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                
            os.remove(path)
    else:
        st.session_state.processed_file_id = None

with tab3:
    st.header("Batch Query Identification")
    uploaded_files = st.file_uploader("Upload multiple audio query clips at once", type=["mp3", "wav"], accept_multiple_files=True, key="batch_upload")
    
    if uploaded_files:
        st.info(f"📋 Total clips loaded in queue: {len(uploaded_files)}")
        
        if st.button("Process Batch Run"):
            
            if len(uploaded_files) > 30:
                st.error("⚠️ **Server Memory Protection Active:** You have uploaded more than 30 files. To prevent the free cloud container from running out of RAM, please remove some files and ensure you process a maximum of 10 files at a time.")
            else:
                
                import gc
                import os
                
                status_table = st.empty()
                batch_output = []
                progress_bar = st.progress(0)
                
                for idx, f in enumerate(uploaded_files):
                    p = None
                    try:
                        p = save_temp_file(f)
                        q_hashes, _, _, _ = back.extract_features(p)
                        
                        b_votes = defaultdict(lambda: defaultdict(int))
                        for q in q_hashes:
                            matches = back.index.get((q[0], q[1], q[2]))
                            if matches:
                                for song, t_song in matches:
                                    offset = t_song - q[3]
                                    b_votes[song][offset] += 1
                                    
                        predicted = "None"
                        if b_votes:
                            predicted = max(b_votes, key=lambda s: max(b_votes[s].values()))
                            if predicted.lower().endswith(('.mp3', '.wav')):
                                predicted = predicted[:-4]
                            
                        clean_filename = os.path.splitext(f.name)[0]
                        batch_output.append({"Filename": clean_filename, "Prediction": predicted})
                        
                    except Exception as e:
                        batch_output.append({"Filename": f.name, "Prediction": "Error"})
                        
                    finally:
                        if p and os.path.exists(p):
                            os.remove(p)
                        gc.collect()
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    df_current = pd.DataFrame(batch_output)
                    status_table.dataframe(df_current, width="stretch", hide_index=True)
                
                csv_data = df_current.to_csv(index=False).encode('utf-8')
                st.success("✅ Batch run completed successfully!")
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="batch_identification_results.csv",
                    mime="text/csv"
                )