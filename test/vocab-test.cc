#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <memory>
#include <string>

#include "../mse/rawtext-splitter.h"
#include "../mse/vocab.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(vocab_test)

BOOST_AUTO_TEST_CASE(constructor_and_setter_test) {
  string punctuation = "/⁄\\()\"':,.;<>~!@#$%^&*|+=[]{}`?-…-‹›«»‐-‘’“”";
  auto text_splitter = make_shared<RawtextSplitter>(punctuation);
  Vocab vocab("data/test", 2000, text_splitter);
  BOOST_CHECK_EQUAL(0, vocab.id("<UNK>"));
  BOOST_CHECK_EQUAL(1, vocab.id("<s>"));
  BOOST_CHECK_EQUAL(2, vocab.id("</s>"));
  BOOST_CHECK_EQUAL(3, vocab.id("the"));
  BOOST_CHECK_EQUAL(4, vocab.id(","));
  BOOST_CHECK_EQUAL(5, vocab.id("."));
  BOOST_CHECK_EQUAL(6, vocab.id("of"));
  BOOST_CHECK_EQUAL(7, vocab.id("and"));
  BOOST_CHECK_EQUAL(8, vocab.id("\""));
  BOOST_CHECK_EQUAL(9, vocab.id("in"));
  BOOST_CHECK_EQUAL("<UNK>", vocab.word(0));
  BOOST_CHECK_EQUAL("<s>", vocab.word(1));
  BOOST_CHECK_EQUAL("</s>", vocab.word(2));
  BOOST_CHECK_EQUAL("the", vocab.word(3));
  BOOST_CHECK_EQUAL(",", vocab.word(4));
  BOOST_CHECK_EQUAL(".", vocab.word(5));
  BOOST_CHECK_EQUAL("of", vocab.word(6));
  BOOST_CHECK_EQUAL("and", vocab.word(7));
  BOOST_CHECK_EQUAL("\"", vocab.word(8));
  BOOST_CHECK_EQUAL("in", vocab.word(9));
  BOOST_CHECK_EQUAL(322, vocab.frequency(0));
  BOOST_CHECK_EQUAL(109, vocab.frequency(1));
  BOOST_CHECK_EQUAL(109, vocab.frequency(2));
  BOOST_CHECK_EQUAL(654, vocab.frequency(3));
  BOOST_CHECK_EQUAL(489, vocab.frequency(4));
  BOOST_CHECK_EQUAL(411, vocab.frequency(5));
  BOOST_CHECK_EQUAL(380, vocab.frequency(6));
  BOOST_CHECK_EQUAL(324, vocab.frequency(7));
  BOOST_CHECK_EQUAL(270, vocab.frequency(8));
  BOOST_CHECK_EQUAL(260, vocab.frequency(9));
  BOOST_CHECK_EQUAL(2003, vocab.size());
  BOOST_CHECK_EQUAL(10135, vocab.num_words());
}

BOOST_AUTO_TEST_SUITE_END()
