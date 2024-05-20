import gmpy2
from gmpy2 import invert, powmod

def get_field(bitlength):
    if bitlength <= 64:
        P64Field.bits = bitlength
        field = P64Field
    elif bitlength <= 128:
        P128Field.bits = bitlength
        field = P128Field
    elif bitlength <= 256:
        P256Field.bits = bitlength
        field = P256Field
    elif bitlength <= 512:
        P512Field.bits = bitlength
        field = P512Field
    elif bitlength <= 1024:
        P1024Field.bits = bitlength
        field = P1024Field
    elif bitlength <= 2048:
        P2048Field.bits = bitlength
        field = P2048Field
    else:
        raise ValueError("No sufficient field for this secret size")
    return field

class PField(object):
    def __init__(self, encoded_value, p, bits):
        self.p = p
        self.bits = bits
        if isinstance(encoded_value, gmpy2.mpz):
            self._value = encoded_value
        elif isinstance(encoded_value, int):
            self._value = gmpy2.mpz(encoded_value)
        elif isinstance(encoded_value, bytes):
            self._value = gmpy2.mpz(int.from_bytes(encoded_value, "big"))
        else:
            raise ValueError(
                "The encoded value is of type {} but it must be an integer or a byte string".format(
                    type(encoded_value)
                )
            )

    def __eq__(self, other):
        return self._value == other._value

    def __int__(self):
        return self._value

    def __hash__(self):
        return self._value.__hash__()

    def encode(self):
        return self._value.to_bytes(256, "big")

    def __mul__(self, factor):
        return PField((self._value * factor._value) % self.p, self.p, self.bits)

    def __add__(self, term):
        return PField((self._value + term._value) % self.p, self.p, self.bits)

    def __sub__(self, term):
        return PField((self._value - term._value) % self.p, self.p, self.bits)

    def __div__(self, term):
        # use the inverse
        return PField(
            (self._value * invert(term._value, self.p)) % self.p, self.p, self.bits
        )

    def inverse(self):
        if self._value == 0:
            raise ValueError("Inversion of zero")

        return PField(invert(self._value, self.p), self.p, self.bits)

    def __pow__(self, exponent):
        return PField(powmod(self._value, exponent._value, self.p), self.p, self.bits)

    def __mod__(self, mod):
        return PField(self._value % mod._value, self.p, self.bits)

    def get_real_size(self):
        return (self._value.bit_length() + 7) // 8

    def __repr__(self):
        return self._value.__repr__()


class P2048Field(PField):
    bits = 2048

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**2203 - 1, P2048Field.bits)


class P1024Field(PField):
    bits = 1024

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**1279 - 1, P1024Field.bits)


class P512Field(PField):
    bits = 512

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**521 - 1, P512Field.bits)


class P256Field(PField):
    # 2**n - k
    bits = 256

    def __init__(self, encoded_value):
        super().__init__(
            encoded_value,
            115792089210356248762697446949407573529996955224135760342422259061068512044369,
            P256Field.bits,
        )


class P128Field(PField):
    # 2**n - k
    bits = 128

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**129 - 1365, P128Field.bits)


class P64Field(PField):
    # 2**n - k
    bits = 64

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**65 - 493, P64Field.bits)


def get_two_safe_primes(bitlength):
    if bitlength == 1024:
        p = 95201490171292091305343691901536093022951520003576908941900149825029005389867
        q = 89883304695025323233172318520056871790432552701138926681543216001826734635419
    elif bitlength == 2048:
        p = 7612036456999683599077717132671455415031908407114945567931900462632365597206721501244304073075221192967828168004574853215746650933832090986220158139468003
        q = 9952622069270053370380344155062236088413830836754885113824514979258221546321394097768536763954080795280385430886254765458716253805918045968364867081064203
    elif bitlength == 4096:
        p = 156171196023471764965930038068236212130742833720153106835167216676997394968416508970275487227265027714772581140817321565789192429799091434620901203089139350978272599315268832422099774313791790176366729425577463186294817143877359666951633393929590150360560691713479213994634278538260551525751878787457936150167
        q = 175791022837015142645179339814548331030891265255834441521864048307932186591750660041545905708601370557207685171431174852277287144826585032161845199188668231553079242457012929889570445049558227948536468242116083725345169686579402487015565178485960854816230484108917304927570171054986177479328522459951938194267
    else:
        raise ValueError("bitlength should be 2048 or 4096")
    return p, q

def get_predefined_parameters(order2):
    predefined_params = {
        64: (118993496723605720649593660529207168864733744326723896923567962318969856712908223695782903376018610660779623741548780418365956041330447543466155856162977294112387881310113640813531592181477826621222551038116218711300286077356395979786912489720496239299355824734316899586457257092129893858736368407556123251429086245812955711765295727967098244939707088620567109172277300852565567280503106022545291760465530324187173079497220775584425365329497507306171046845213400282100828642269594540574548058035141799648435116114806677788038612285642336956862458195520452474371021785682441568811347292883237532781961293635757813240587393, 
             65921088095567578269704299870714368857565450269598693821094900791300352883896301677937293422899170267750937550212026130105988749091391058070376164399978100880125805838459489005406937039943464741982353099547305991875012438697159586909147218322799082565925699555476515301840914229244724702802073231306838041396280150671491701746742593030962783607018294306176574577342348893760959757385098056314210505279608753409673321692672382397133698912851398895958674024761108882994495566261659107117630319756158222617501736319686616540104280489373127601829549085974519720611717078848070264014803609680409105837440455790456504353449479),
        128: (106981010039837340760118745375683519737891450245152873902815817262969625473512939634166177888071073435375196126977811205112056925032305565840668520904261571911102737537244877727283321475749479730572533905954656679611604235617002469902973475147139428390318190799928875013775302833008813043408751792966558805826454031456364588292599463527751613777306781622214032353857242880564081366689623146375358989721314551336913115716464724295317784774400909555638161408816079663569595999099110329810133561955850278553630840774216779605624170673249413147020923852076920653293251654008930433914555296790601508867134566085024867456424577, 
             36318915013307881068770167656769930768366335322241498476359586405876629134903032480607589087494611014457741032796171982706791801588857005643149438315179078623738262023889498730848378781953551453449723884114203884583098297812839648153331217998271374521442296462058034971986620523138687611280632705291459337291763559012808661330757117076134162795019320033533206048461756925793319467511334907606608975915756884082352550195608422404483404928956688840171209133336035107596329157432303614391502167613600372004158400848999860441535180321360805810649166444587842720425972930197049695584544874606841881558645315659944116726873725),
        256: (297709916699103944768289003547885539480153111935956400237281781374912074000652457287182560413507424878423157106257432247664497794868045694487911443207133491879575294160361724186486909386413249149850772280930266829500054855401255528572104538737389373457549949187573705149203384448136886655039798195309220226364518240417713731886804683668658377463379715968698532449159923483226142164307355599556237961258951342236645398880103828332613475162813361000433928876572850741887504306462283803550137244206678417076856964551761759098797541788820482795310195544369091979206183518580049001814365049878859416896432708305674910605827329, 
              233083103940744302009782052289819615072402726415559233823810952260931497073592255050745510515674358961142131693830003204074136576065052222622917284579233593890223090148791957106728346580802649056143917755957412178361958450081320217068876859232257387317045316376459500528352998541819414831712689406133164537453216323477798475996622798540870815616875150677191565588343396457323917679702658832102962763123598365914323313052719266262182697863411816155583809754215053429201126286532053643189186811024568391190349706316916591037034773369676395208973815669682429548751139758897587427511478768393693360565839896540317040800109936),
        512: (998635410850751563484702556627856887297899803045744598794560465192975130307182463486001744906280823718199703730041117128954329079628401833281480320400421244034291032568762067533868042311646754968236451273831478949972356105031825446054791328458219583607808310086306974643537536207285746237161507858729355678595765310356550830252520935836962813690949953236946504659492118269916855650758306448071489293154637786398529985271720531070653454076910640704292291500870763766878079953029116440627924137193546278036445198616150666672288615447992369577124108684294025361608759776568823000585446505425820529655194418121543500098041857,
                173379136947097250992483191948712119965063893081101819072169666852275989136409153137129230629430832655560816645771777068710790399365141033725559654778988188309509451408230895754539914319503975679778936490813499333685844810373978082449931911470046794873205174315126403071343741332560927610590092633787713675786417354925832303312091825108569619759920328496624899483507900213154758747978795259733643625127790082323583216738668673687895821958396436809856473594206255917617521964233888721326797594425297222406884778301035855224717212782101499121574033148214197179419347629899969505632491756115553049931769422840047099023371041)
    }
    return predefined_params[order2]

SMALL_PRIMES = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
    373,
    379,
    383,
    389,
    397,
    401,
    409,
    419,
    421,
    431,
    433,
    439,
    443,
    449,
    457,
    461,
    463,
    467,
    479,
    487,
    491,
    499,
    503,
    509,
    521,
    523,
    541,
    547,
    557,
    563,
    569,
    571,
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
    701,
    709,
    719,
    727,
    733,
    739,
    743,
    751,
    757,
    761,
    769,
    773,
    787,
    797,
    809,
    811,
    821,
    823,
    827,
    829,
    839,
    853,
    857,
    859,
    863,
    877,
    881,
    883,
    887,
    907,
    911,
    919,
    929,
    937,
    941,
    947,
    953,
    967,
    971,
    977,
    983,
    991,
    997,
]
