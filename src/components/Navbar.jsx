import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

function Navbar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  const isActive = (path) => location.pathname === path;
  const closeMenu = () => setOpen(false);

  return (
    <nav className="navbar">
      <Link to="/" className="logo">
        <span className="logo-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="20" height="20">
            <circle cx="12" cy="12" r="2.4" className="logo-node logo-node-center" />
            <circle cx="4" cy="6" r="1.8" className="logo-node" />
            <circle cx="20" cy="6" r="1.8" className="logo-node" />
            <circle cx="4" cy="18" r="1.8" className="logo-node" />
            <circle cx="20" cy="18" r="1.8" className="logo-node" />
            <line x1="12" y1="12" x2="4" y2="6" className="logo-bond" />
            <line x1="12" y1="12" x2="20" y2="6" className="logo-bond" />
            <line x1="12" y1="12" x2="4" y2="18" className="logo-bond" />
            <line x1="12" y1="12" x2="20" y2="18" className="logo-bond" />
          </svg>
        </span>
        <span className="logo-text">
          Antibody<span className="logo-accent">ML</span>
        </span>
      </Link>

      <div className={`nav-links ${open ? "open" : ""}`}>
        <Link to="/" className={isActive("/") ? "active" : ""} onClick={closeMenu}>
          Home
        </Link>
        <Link
          to="/analysis"
          className={isActive("/analysis") ? "active" : ""}
          onClick={closeMenu}
        >
          Analyze
        </Link>
        <Link
          to="/results"
          className={isActive("/results") ? "active" : ""}
          onClick={closeMenu}
        >
          Results
        </Link>
      </div>

      <button
        className={`nav-toggle ${open ? "open" : ""}`}
        aria-label={open ? "Close navigation menu" : "Open navigation menu"}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span></span>
        <span></span>
        <span></span>
      </button>
    </nav>
  );
}

export default Navbar;
