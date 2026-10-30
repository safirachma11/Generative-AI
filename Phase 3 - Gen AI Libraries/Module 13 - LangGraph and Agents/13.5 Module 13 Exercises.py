import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Exercise 1: Step-limit ReAct check (Mencegah loop tak terhingga)
def check_iteration_limit(step_count: int, max_steps: int = 8) -> bool:
    """Return True if step_count has reached or exceeded max_steps."""
    return step_count >= max_steps

# Exercise 2: Simulated Safe ReAct Guardrail
def run_react_guardrail_demo():
    max_allowed_steps = 5
    current_step = 0
    print(f"Batas Maksimum Iterasi (Guardrail): {max_allowed_steps} langkah")
    
    while current_step < 7:
        current_step += 1
        is_limit_reached = check_iteration_limit(current_step, max_allowed_steps)
        if is_limit_reached:
            print(f" [Langkah {current_step}]: STOP! Iterasi mencapai batas maksimum ({max_allowed_steps}). Mencegah infinite loop.")
            break
        else:
            print(f" [Langkah {current_step}]: Agen mengeksekusi aksi... (Aman)")

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 13 EXERCISES - ReAct Guardrails")
    print("=" * 60)
    
    print("\n--- Uji Logika Guardrail Limit ---")
    print("Uji langkah 4 dari 8 : Batas tercapai?", check_iteration_limit(4, 8))
    print("Uji langkah 9 dari 8 : Batas tercapai?", check_iteration_limit(9, 8))
    
    print("\n--- Simulasi Loop ReAct dengan Guardrail ---")
    run_react_guardrail_demo()
    
