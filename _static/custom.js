// Custom breakpoint for sidebar toggles (1366px instead of default ~960px)
document.addEventListener('DOMContentLoaded', function() {
  const CUSTOM_BREAKPOINT = 1366;

  function setupCustomToggles() {
    // Primary sidebar toggle
    const primaryToggle = document.querySelector('.sidebar-toggle.primary-toggle');
    const primarySidebar = document.getElementById('pst-primary-sidebar');
    const primaryModal = document.getElementById('pst-primary-sidebar-modal');

    // Secondary sidebar toggle
    const secondaryToggle = document.querySelector('.sidebar-toggle.secondary-toggle');
    const secondarySidebar = document.getElementById('pst-secondary-sidebar');
    const secondaryModal = document.getElementById('pst-secondary-sidebar-modal');

    // Override primary toggle click
    if (primaryToggle && primarySidebar) {
      primaryToggle.addEventListener('click', function(e) {
        if (window.innerWidth <= CUSTOM_BREAKPOINT && window.innerWidth > 960) {
          e.preventDefault();
          e.stopPropagation();
          primarySidebar.classList.toggle('show');
          const isOpen = primarySidebar.classList.contains('show');
          primaryToggle.setAttribute('aria-expanded', isOpen);
        }
      }, true);
    }

    // Override secondary toggle click
    if (secondaryToggle && secondarySidebar) {
      secondaryToggle.addEventListener('click', function(e) {
        if (window.innerWidth <= CUSTOM_BREAKPOINT && window.innerWidth > 960) {
          e.preventDefault();
          e.stopPropagation();
          secondarySidebar.classList.toggle('show');
          const isOpen = secondarySidebar.classList.contains('show');
          secondaryToggle.setAttribute('aria-expanded', isOpen);
        }
      }, true);
    }

    // Close sidebars when clicking outside
    document.addEventListener('click', function(e) {
      if (window.innerWidth <= CUSTOM_BREAKPOINT && window.innerWidth > 960) {
        if (primarySidebar && primarySidebar.classList.contains('show')) {
          if (!primarySidebar.contains(e.target) && !primaryToggle.contains(e.target)) {
            primarySidebar.classList.remove('show');
            primaryToggle.setAttribute('aria-expanded', 'false');
          }
        }
        if (secondarySidebar && secondarySidebar.classList.contains('show')) {
          if (!secondarySidebar.contains(e.target) && !secondaryToggle.contains(e.target)) {
            secondarySidebar.classList.remove('show');
            secondaryToggle.setAttribute('aria-expanded', 'false');
          }
        }
      }
    });
  }

  setupCustomToggles();
});