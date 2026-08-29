// Operator review queue: capture a pass/fail validity label for a stored
// run/iteration/task without exposing generated HTML through the JSON API.
(() => {
  const TOKEN_KEY = 'design_gan_start_token';
  const tokenInput = document.getElementById('review-token');
  if (tokenInput) {
    tokenInput.value = localStorage.getItem(TOKEN_KEY) || '';
    tokenInput.addEventListener('change', () => {
      if (tokenInput.value) localStorage.setItem(TOKEN_KEY, tokenInput.value);
      else localStorage.removeItem(TOKEN_KEY);
    });
  }

  document.querySelectorAll('.review-form button[data-label]').forEach((button) => {
    button.addEventListener('click', async () => {
      const card = button.closest('.review-card');
      const form = button.closest('.review-form');
      const status = form.querySelector('.review-status');
      const caseInput = form.querySelector('input[name="case_id"]');
      const buttons = form.querySelectorAll('button');
      const token = (tokenInput && tokenInput.value) || localStorage.getItem(TOKEN_KEY);
      const payload = {
        run_id: Number(card.dataset.runId),
        iteration: Number(card.dataset.iteration),
        task_id: card.dataset.taskId,
        case_id: caseInput.value.trim(),
        expected_pass: button.dataset.label === 'pass',
      };
      const headers = { 'content-type': 'application/json' };
      if (token) {
        headers.Authorization = `Bearer ${token}`;
        localStorage.setItem(TOKEN_KEY, token);
      }
      buttons.forEach((item) => { item.disabled = true; });
      status.textContent = 'saving label…';
      try {
        const response = await fetch('/api/evaluator-cases', {
          method: 'POST', headers, body: JSON.stringify(payload),
        });
        if (!response.ok) {
          if (response.status === 401) localStorage.removeItem(TOKEN_KEY);
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || response.statusText);
        }
        const saved = await response.json();
        form.innerHTML = `<p class="review-captured">Captured <code>${escapeHtml(saved.id)}</code>
          as <b>${saved.expected_pass ? 'should pass' : 'should fail'}</b>.</p>`;
      } catch (error) {
        status.textContent = `error: ${error.message}`;
        buttons.forEach((item) => { item.disabled = false; });
      }
    });
  });

  function escapeHtml(value) {
    const node = document.createElement('span');
    node.textContent = value;
    return node.innerHTML;
  }
})();
