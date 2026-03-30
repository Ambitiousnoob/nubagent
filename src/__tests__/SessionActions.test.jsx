import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { SessionActions } from '../components/Library/SessionActions.jsx';

describe('SessionActions', () => {
  it('saves edited session fields', () => {
    const onEdit = vi.fn();
    const onClose = vi.fn();

    render(
      <SessionActions
        isOpen
        session={{
          id: 'session-1',
          query: 'Original query',
          heading: 'Original heading',
          body: 'Original body',
        }}
        onEdit={onEdit}
        onClose={onClose}
      />
    );

    fireEvent.click(screen.getByText('Edit Session'));
    fireEvent.change(screen.getByLabelText('Query'), { target: { value: 'Updated query' } });
    fireEvent.change(screen.getByLabelText('Heading'), { target: { value: 'Updated heading' } });
    fireEvent.change(screen.getByLabelText('Body'), { target: { value: 'Updated body' } });
    fireEvent.click(screen.getByText('Save changes'));

    expect(onEdit).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'session-1' }),
      {
        query: 'Updated query',
        heading: 'Updated heading',
        body: 'Updated body',
      }
    );
    expect(onClose).toHaveBeenCalled();
  });
});
