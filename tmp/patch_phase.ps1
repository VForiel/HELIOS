$old = @'
    # Debug: check result structure
    st.write("### Debug Info")
    st.write(f"Result keys: {list(result.keys())}")
    st.write(f"Has 'metric' key: {'metric' in result}")
    
    # Plot convergence history
    if "metric" in result:
        metric_array = np.asarray(result["metric"], dtype=float)
        st.write(f"Metric array shape: {metric_array.shape}, dtype: {metric_array.dtype}")
        st.write(f"Metric array length: {len(metric_array)}")
        st.write(f"First 5 values: {metric_array[:5] if len(metric_array) >= 5 else metric_array}")
        
        if len(metric_array) > 0:
            st.write("Creating plot...")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.semilogy(metric_array, "o-", lw=2, markersize=6, color="steelblue")
            ax.axhline(y=result["best_metric"], color="red", linestyle="--", lw=2, label=f"Best: {result['best_metric']:.3e}")
            ax.set_xlabel("Iteration", fontsize=11)
            ax.set_ylabel("Null Depth Metric (null/bright)", fontsize=11)
            ax.set_title("Phase Calibration Convergence", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            st.write("Displaying plot...")
            st.pyplot(fig, use_container_width=True)
            st.write("Plot displayed ✓")
            plt.close(fig)
        else:
            st.warning("Metric array is empty!")
    else:
        st.error("'metric' key not found in result!")

    # Defer updating widget-bound keys to the next run to avoid Streamlit key mutation errors
    st.session_state["phase_updates"] = [float(phi / np.pi) for phi in result["best_phases"]]
    st.rerun()
'@

$new = @'
    # Plot convergence history
    if "metric" in result:
        metric_array = np.asarray(result["metric"], dtype=float)
        if len(metric_array) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.semilogy(metric_array, "o-", lw=2, markersize=6, color="steelblue")
            ax.axhline(y=result["best_metric"], color="red", linestyle="--", lw=2, label=f"Best: {result['best_metric']:.3e}")
            ax.set_xlabel("Iteration", fontsize=11)
            ax.set_ylabel("Null Depth Metric (null/bright)", fontsize=11)
            ax.set_title("Phase Calibration Convergence", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        else:
            st.warning("Metric array is empty; no convergence plot to display.")
    else:
        st.warning("Result does not contain a 'metric' history; no plot to display.")

    # Stage updates and offer manual apply to avoid immediate rerun clearing the plot
    st.session_state["phase_updates"] = [float(phi / np.pi) for phi in result["best_phases"]]
    st.info("Calibrated phases are staged. Click 'Apply calibrated phases' to update sliders.")
    if st.button("Apply calibrated phases", key="apply_phases_btn"):
        st.rerun()
'@

$path = 'examples/14_mmi_streamlit.py'
(Get-Content $path -Raw) -replace [regex]::Escape($old), $new | Set-Content -Encoding UTF8 $path
Write-Host 'phase block updated'
