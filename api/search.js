const google = require('googlethis');

module.exports = async (req, res) => {
  const query = req.query.q;

  if (!query) {
    return res.status(400).json({ error: 'Query parameter "q" is required' });
  }

  try {
    const { results } = await google.search(query, { page: 0, safe: false });
    res.status(200).json(results);
  } catch (error) {
    console.error('Google search failed:', error);
    res.status(500).json({ error: 'Failed to perform Google search' });
  }
};
