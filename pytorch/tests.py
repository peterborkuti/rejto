from generate import Util

assert "" == Util.decode_char_tensor(Util.char_tensor(""))
assert "x" == Util.decode_char_tensor(Util.char_tensor("x"))
assert "hello" == Util.decode_char_tensor(Util.char_tensor("hello"))


from generate import Train

def learn_character_set(trainString, maxSteps = 100, print_every=10):
    print(trainString)
    primeStr = trainString[:1]
    train = Train(file='')

    beforeString = train.decoder.generate(prime_str=primeStr)
    # first guess should be totally random but printable characters
    assert len(set(beforeString)) > len(beforeString) / 2

    testString = " "
    step = 0
    prevLoss = 1000
    loss = 999
    while loss < prevLoss and testString != trainString and maxSteps > step:
        loss = train.train(trainString)
        testString = train.decoder.generate(prime_str=primeStr, predict_len=len(trainString) - 1, temperature=0.6
        )
        step += 1
        #if step % print_every == 0:
        print (step, testString) 

    return step, testString

def test_string(string, steps, print_every=10):
    step, s = learn_character_set(string, maxSteps=steps, print_every=print_every)
    assert step <= steps, str(step) + ">" + str(steps)
    assert s == string, s
    print(string + " learned in " + str(step) + " steps")

import unidecode

test_string("a" * 20, 3)
# test_string("b" * 20, 3)
# test_string("ab " * 20, 50)
# test_string("alma " * 20, 200)
# test_string("Indul a gorog aludni. " * 20, 200)
# test_string(unidecode.unidecode(
#    "Aki Curzon urat látta a hajón reggelizni (sült sonka tojással, tea, vaj, pirított zsemle)," * 5), 200)



rejto = unidecode.unidecode("""
Aki Curzon urat látta a hajón reggelizni (sült sonka tojással, tea, vaj, pirított zsemle), nem mondta volna róla, hogy ez az úr elõzõ napon húszezer dollárt sikkasztott. Abban a nyugalomban, amellyel a vajat a zsemlére kente, a vaj fölé elhelyezte a sült sonkát, a felsoroltakat leharapta, és mielõtt hozzáfogott volna a rágáshoz, az egészet leöntötte két korty teával, ebben az elmélyült, gondos táplálkozásban benne volt az önmagával megelégedett nagytõkés kényelmes, bölcs, harmonikus életrendszerének egy jellegzetes mozaikja.
Csak azért hangsúlyozzuk ki annyira ezt a különben elég közönséges körülményt egy ember napi életében, mert Curzon úr életében elõször evett ilyen bõségesen a kora reggeli órákban. Teája mellé nélkülözte eddig a tojást és a sonkát.
Mióta tisztviselõ lett, tehát tíz éven át, a zsemlét vajjal bár, de pirítatlanul fogyasztotta, mert mielõtt irodába ment, jóformán arra sem volt elég ideje, hogy a legegyszerûbb reggelit nyugodtan fogyassza el.
Most, miután sikkasztott, úgy érezte alkalma és ideje van az életet nyugodtabb szemlélõdéssel tanulmányozni és ezen a bûnügyi tanulmányútján elsõ experimentuma az imént megénekelt villásreggeli volt. Curzon úr, e regényünk hõse, sajnos, nélkülözte mindazon kvalifikációkat, amelyek alkalmasak arra, hogy egy embert regényhõssé tegyenek. Mind a két szóval szöges ellentétben állt az egyénisége. A regényhez csak igen kevés köze volt, ilyet nem olvasott, és nem írt, eseménytelen életét elkerülték a nagy szerelmek, sohasem párbajozott, egyáltalán semmit sem csinált, ami akár csak egy rövid lélegzetû elbeszélésre is anyagot adhatott volna. Ami a hõst illeti, Curzon úr átment az utca másik oldalára, ha két ember hangosan vitatkozott, és bár szégyellte bevallani, titokban félt a részeg emberektõl. És miután nem volt regényhõs, nevezzük Curzon urat a regény gyávájának.
Az olvasónak igaza lesz, ha ezek után azt mondja; hohó, egy ember, aki gyáva, aki irtózik minden veszélytõl, kalandtól akit úgy ábrázol a regény elsõ néhány sora, mint a legfantáziátlanabb nyárspolgárt, hogy lehet az sikkasztó? Hogy kerülhet egy ilyen hajóra, ilyen helyzetbe? Egy villásreggelivel!
Hallgassák meg, kérem a sikkasztás történetét, azután az vesse rám az elsõ követ, aki nem lusta hozzá."
""")

#test_string(rejto * 5, 2000, print_every=100)