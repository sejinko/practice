# 리스트(다른 객체들로 구성된 특별한 형태의 벡터)

# 리스트 생성하기

x <- list(1, 'a', TRUE, 1+4i)
x

Hong <- list(kor_name = '홍길동', eng.name='gil-dong',
             age=43, marrie=T, no.child=2, child.ages=c(13,10))
Hong

# 객체의 자료구조를 요약해서 보여줌
str(Hong)

# 리스트의 각 성분에 접근하기

Hong$kor_name

Hong$kor_name[1]

Hong$child.ages

Hong$child.ages[2]


Hong['age']

Hong['child.ages']

Hong['child.ages'][1] # 이건 왜 안되지?

Hong[c(1,2)]



x <- list(a = 1:10,
          beta = exp(-3:3),
          logic = c(TRUE, FALSE, FALSE, TRUE))
x














