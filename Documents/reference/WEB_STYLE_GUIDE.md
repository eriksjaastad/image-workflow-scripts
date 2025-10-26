# Web Interface Style Guide
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26

**Erik's Image Processing Workflow - Consistent Design System**

## 🎨 Core Color Palette

### Primary Colors
```css
:root {
  color-scheme: dark;
  --bg: #101014;           /* Main background - deep navy */
  --surface: #181821;      /* Card/panel backgrounds */
  --surface-alt: #1f1f2c;  /* Alternative surface (darker) */
  --accent: #4f9dff;       /* Primary blue - friendly & professional */
  --accent-soft: rgba(79, 157, 255, 0.2); /* Transparent accent */
}
```

### Action Colors
```css
--success: #51cf66;        /* Green - confirmations, success states */
--danger: #ff6b6b;         /* Red - warnings, delete actions */
--warning: #ffd43b;        /* Yellow/Orange - cautions, secondary actions */
--muted: #a0a3b1;          /* Gray - secondary text, disabled states */
```

### Semantic Usage
- **Blue (`--accent`)**: Primary actions, selected states, main buttons
- **White**: Crop/edit actions, special selections, high contrast
- **Green (`--success`)**: Confirmations, completed actions
- **Red (`--danger`)**: Delete hints, destructive actions
- **Yellow (`--warning`)**: Secondary actions, batch info, cautions
- **Gray (`--muted`)**: Supporting text, metadata, disabled states

## 🔘 Button System

### Button Types & Colors

#### Primary Action Buttons
```css
.btn-primary {
  background: var(--accent);      /* Blue */
  color: white;
  border: 2px solid var(--accent);
}

.btn-primary:hover {
  background: #4dd78a;           /* Slight blue tint on hover */
  transform: translateY(-1px);   /* Subtle lift effect */
}
```

#### Selection Buttons (Image Selector)
```css
.action-btn.image-active {
  background: var(--accent);      /* Blue for selected */
  border-color: var(--accent);
  color: white;
}

.action-btn.crop-active {
  background: white;              /* White for crop */
  border-color: white;
  color: black;
}
```

#### Character Sorter Buttons
```css
.btn-process {
  background: linear-gradient(135deg, var(--warning) 0%, #ffb700 100%);
  color: black;                   /* Yellow gradient for processing */
}

.btn-refresh, .btn-next {
  background: var(--accent);      /* Blue for navigation */
  color: white;
}
```

### Button States
- **Idle**: Surface color with subtle border
- **Hover**: Accent color with slight lift (`translateY(-1px)`)
- **Active/Selected**: Full accent color with solid border
- **Disabled**: 50% opacity, no hover effects

## 📐 Spacing & Layout

### Standard Spacing Scale
```css
--space-xs: 0.25rem;    /* 4px - tight spacing */
--space-sm: 0.5rem;     /* 8px - compact elements */
--space-md: 0.75rem;    /* 12px - standard spacing */
--space-lg: 1rem;       /* 16px - comfortable spacing */
--space-xl: 1.5rem;     /* 24px - section breaks */
--space-2xl: 2rem;      /* 32px - major separations */
```

### Padding Standards
- **Buttons**: `0.75rem` (12px) vertical, `1rem-2rem` horizontal
- **Cards/Panels**: `0.75rem` (12px) minimum, `1rem` (16px) comfortable
- **Page Content**: `1rem` (16px) minimum, `2rem` (32px) for major sections
- **Header/Toolbar**: `1rem` (16px) standard

### Border Radius
- **Small elements**: `8px` (buttons, cards)
- **Large panels**: `12px` (image groups, main sections)
- **Subtle**: `4px` (inputs, small components)

## 📱 Layout Patterns

### Sticky Headers
```css
.header {
  position: sticky;
  top: 0;
  background: var(--bg);
  z-index: 100;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}
```

### Card/Panel Design
```css
.card {
  background: var(--surface);
  border-radius: 12px;
  padding: 0.75rem;
  border: 2px solid transparent;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
  border-color: var(--accent-soft);
  box-shadow: 0 8px 16px rgba(0,0,0,0.15);
}
```

### Sidebar Pattern (Image Selector)
```css
.action-sidebar {
  width: 120px;
  background: var(--surface-alt);
  position: fixed;
  right: 0;
  top: 0;
  bottom: 0;
  border-left: 1px solid rgba(255,255,255,0.1);
}
```

## 🖼️ Image Display

### Image Cards
```css
.image-card {
  border-radius: 8px;
  padding: 0.5rem;
  position: relative;
  transition: all 0.3s ease;
}

.image-card img {
  max-width: 100%;
  max-height: 60vh;
  width: auto;
  height: auto;
  object-fit: contain;  /* Never crop images */
  border-radius: 4px;
}
```

### Selection States
```css
.image-card.selected {
  border: 2px solid var(--accent);        /* Blue selection */
  box-shadow: 0 0 0 2px rgba(79,157,255,0.3);
}

.image-card.crop-selected {
  border: 2px solid white;                /* White crop outline */
  box-shadow: 0 0 0 2px rgba(255,255,255,0.6);
}

.image-card.delete-hint {
  border: 2px solid var(--danger);        /* Red delete warning */
  box-shadow: 0 0 0 2px rgba(255,107,107,0.3);
}
```

## 📝 Typography

### Font Stack
```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', system-ui, sans-serif;
```

### Text Hierarchy
- **Headers**: `1rem-1.2rem`, `font-weight: 600`
- **Body**: `0.9rem`, `font-weight: 400`
- **Metadata**: `0.65rem-0.7rem`, `color: var(--muted)`
- **Button Text**: `0.9rem-1rem`, `font-weight: 600`

## 🎯 Interactive Feedback

### Hover Effects
- **Buttons**: Color change + `translateY(-1px)` lift
- **Cards**: Border highlight + subtle shadow
- **Images**: Border color transition

### Transition Standards
```css
transition: all 0.2s ease;        /* Fast interactions */
transition: all 0.3s ease;        /* Comfortable animations */
```

### Focus States
- Use accent color with `box-shadow` for keyboard navigation
- Maintain high contrast for accessibility

## 🔄 Consistency Rules

### Do's ✅
- Use the defined color variables consistently
- Apply standard spacing scale
- Use blue for primary selections
- Use white for crop/edit actions
- Keep button padding consistent (0.75rem vertical)
- Apply subtle hover animations
- Use sticky headers for better navigation

### Don'ts ❌
- Don't use random colors (resist yellow temptations! 😄)
- Don't crop images - always use `object-fit: contain`
- Don't use harsh transitions (> 0.3s)
- Don't mix different button padding sizes
- Don't use opacity for critical feedback (use color/border instead)

## 🛠️ Implementation

### Color Variable Usage
Always use CSS custom properties:
```css
/* Good */
background: var(--accent);
color: var(--muted);

/* Avoid */
background: #4f9dff;
color: #a0a3b1;
```

### Component Consistency
When creating new buttons, cards, or interactive elements:
1. Start with the base patterns above
2. Apply semantic colors based on function
3. Use standard spacing and transitions
4. Test hover and active states

---

**This style guide ensures our image processing tools feel like a cohesive, professional suite while maintaining the friendly, efficient workflow you love!** 🎨✨
